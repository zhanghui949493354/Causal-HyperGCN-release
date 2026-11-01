#!/usr/bin/env python3
"""
Independent external hold-out evaluation.

External-validation workflow:
  1) Train on all curated associations in data/association.csv
  2) Score MDS/ncRNADrug held-out positives after overlap removal
  3) Compute AUC/AUPRC on feature-mappable pairs and summarize coverage

Usage (from repository root):
  python external_validation/run_external_validation.py
  python external_validation/run_external_validation.py --smoke

Requires:
  - data/ (processed features + association.csv)
  - external_validation/external_pairs.csv  (held-out positives; shipped with this folder)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from train import main_new  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = HERE / "external_pairs.csv"
OUT_JSON = HERE / "eval_results.json"


def normalize_drug_key(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name).strip()).lower()
    s = s.replace("_", " ").replace("/", " ")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9\-\s\+]", " ", s).replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def normalize_mirna_key(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name).strip()).lower()
    for p in ("hsa-", "mmu-", "rno-", "mature_"):
        if s.startswith(p):
            s = s[len(p) :]
    s = s.replace("microrna-", "mir-").replace("mirna-", "mir-")
    s = s.replace("*", "-3p").replace("_", "-").replace(" ", "")
    s = s.replace("mir-", "mir").replace("let-7", "let7")
    s = re.sub(r"-+", "-", s)
    m = re.match(r"^mir([a-z0-9]+)(?:-(3p|5p))?$", s)
    if m:
        return f"mir{m.group(1)}-{m.group(2)}" if m.group(2) else f"mir{m.group(1)}"
    if not s.startswith(("mir", "let")):
        s = "mir" + s
    return s


def build_lookups(drug_df: pd.DataFrame, mirna_df: pd.DataFrame):
    drug_by_key, mir_by_key = {}, {}
    for _, row in drug_df.iterrows():
        name = str(row["drug"])
        did = int(row["id"])
        drug_by_key[name.lower()] = (did, name)
        drug_by_key[normalize_drug_key(name)] = (did, name)
    for _, row in mirna_df.iterrows():
        name = str(row["miRNAname"])
        mid = int(row["id"])
        mir_by_key[name] = (mid, name)
        mir_by_key[name.lower()] = (mid, name)
        mir_by_key[normalize_mirna_key(name)] = (mid, name)
        if name.startswith("miR-"):
            mir_by_key["hsa-" + name] = (mid, name)
            mir_by_key[normalize_mirna_key("hsa-" + name)] = (mid, name)
    return drug_by_key, mir_by_key


def map_external_pairs(ext: pd.DataFrame, drug_by_key, mir_by_key, miRNA_num: int):
    mapped = []
    n_drug_miss = n_mir_miss = n_both_miss = 0
    for _, r in ext.iterrows():
        d_ok = m_ok = None
        for cand in [r.get("drug"), r.get("drugbank_id")]:
            if cand is None or (isinstance(cand, float) and np.isnan(cand)):
                continue
            s = str(cand)
            hit = drug_by_key.get(s.lower()) or drug_by_key.get(normalize_drug_key(s))
            if hit:
                d_ok = hit
                break
        for cand in [r.get("mirna"), r.get("mirna_key")]:
            if cand is None or (isinstance(cand, float) and np.isnan(cand)):
                continue
            s = str(cand)
            hit = (
                mir_by_key.get(s)
                or mir_by_key.get(s.lower())
                or mir_by_key.get(normalize_mirna_key(s))
            )
            if hit is None and not s.startswith("hsa-"):
                hit = mir_by_key.get(normalize_mirna_key("hsa-" + s))
            if hit:
                m_ok = hit
                break
        if d_ok and m_ok:
            mid, _ = m_ok
            did, _ = d_ok
            mapped.append((mid, did + miRNA_num))
        elif (not d_ok) and (not m_ok):
            n_both_miss += 1
        elif not d_ok:
            n_drug_miss += 1
        else:
            n_mir_miss += 1
    return (
        np.array(mapped, dtype=np.int64) if mapped else np.zeros((0, 2), dtype=np.int64),
        {
            "n_external_total": int(len(ext)),
            "n_mappable_both": int(len(mapped)),
            "n_missing_drug_only": int(n_drug_miss),
            "n_missing_mirna_only": int(n_mir_miss),
            "n_missing_both": int(n_both_miss),
            "coverage_both_pct": float(100.0 * len(mapped) / max(len(ext), 1)),
        },
    )


def main():
    ap = argparse.ArgumentParser(description="Independent MDS/ncRNADrug external hold-out")
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS, help="Held-out positive pairs CSV")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--smoke", action="store_true", help="Fast smoke run (20 epochs, MLP)")
    ap.add_argument("--predictor", default="xgboost", choices=["xgboost", "mlp"])
    args = ap.parse_args()

    if not args.pairs.exists():
        raise SystemExit(
            f"Missing {args.pairs}. Place the held-out pair list next to this script "
            "(default: independent_external/external_pairs.csv)."
        )

    drug_df = pd.read_csv(REPO / "data" / "drug.csv")
    mirna_df = pd.read_csv(REPO / "data" / "miRNA.csv")
    miRNA_num = len(mirna_df)
    ext = pd.read_csv(args.pairs)
    mapped, cov = map_external_pairs(ext, *build_lookups(drug_df, mirna_df), miRNA_num)

    print("=== coverage ===")
    print(json.dumps(cov, indent=2, ensure_ascii=False))
    if len(mapped) < 50:
        raise SystemExit("Too few mappable external pairs; abort.")

    epochs = 20 if args.smoke else args.epochs
    predictor = "mlp" if args.smoke else args.predictor
    params = dict(
        fusion_method="multiply_residual",
        gnn_type="causal_hyper",
        predictor_type=predictor,
        negative_sampling_method="kmeans_fractal_cosine",
        cv_type="external",
        max_meta_path_length=4,
        gnn_layers=4,
        hidden_dim=512,
        num_prototypes=25,
        dropout=0.4,
        num_epochs=epochs,
        use_marginalization=True,
        enable_bias_sparsity_report=False,
        external_eval=True,
        external_test_pairs=mapped,
        seed=42,
    )
    print(f"\n=== train + score (epochs={epochs}, predictor={predictor}) ===")
    metrics = main_new(**params) or {}

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "protocol": "P2_MDS_ncRNADrug_heldout",
        "coverage": cov,
        "model": {
            "gnn_type": "causal_hyper",
            "predictor": predictor,
            "hidden_dim": 512,
            "gnn_layers": 4,
            "num_prototypes": 25,
            "max_meta_path_length": 4,
            "dropout": 0.4,
            "negative_sampling_method": "kmeans_fractal_cosine",
            "seed": 42,
            "epochs": epochs,
        },
        "metrics": metrics,
        "eval_status": "DONE" if metrics.get("auc", 0) > 0 else "FAILED_OR_EMPTY",
        "note": (
            "AUC/AUPRC are on the mappable subset only "
            f"(n_pos={cov['n_mappable_both']})."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== DONE ===")
    print(json.dumps({"auc": metrics.get("auc"), "auprc": metrics.get("auprc"), "out": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
