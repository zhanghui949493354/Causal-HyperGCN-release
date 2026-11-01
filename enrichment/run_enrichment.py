#!/usr/bin/env python3
"""
Hypergraph-community enrichment analysis.

Enrichment-analysis workflow:
  1) Rebuild KMeans hyperedges on fused node features (same construction as training)
  2) Map miRNA-rich communities to multi-hit target genes
  3) Run Enrichr (KEGG_2021_Human; optional GO) and write top terms / a simple figure

Usage (from repository root):
  pip install gseapy
  python enrichment/run_enrichment.py
  python enrichment/run_enrichment.py --top_n 5 --fdr 0.05

Requires:
  - data/ (features + miRNA_gene_matrix.csv)
  - network access for Enrichr (gseapy)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from feature_fusion import load_features_and_concatenate  # noqa: E402

HERE = Path(__file__).resolve().parent


def normalize_mirna_name(name: str) -> str:
    s = str(name).strip().lower()
    for p in ("hsa-", "mmu-", "rno-"):
        if s.startswith(p):
            s = s[len(p) :]
    return s.replace("mirna", "mir").replace("mir-", "mir-").replace("*", "").replace(" ", "")


def load_mirna_gene_union_maps(data_dir: Path, mirna_names):
    path = data_dir / "miRNA_gene_matrix.csv"
    df = pd.read_csv(path, index_col=0)
    genes = df.columns.tolist()
    source = {normalize_mirna_name(idx): i for i, idx in enumerate(df.index)}
    mat = np.zeros((len(mirna_names), len(genes)), dtype=np.float32)
    for i, name in enumerate(mirna_names):
        key = normalize_mirna_name(name)
        if key in source:
            mat[i] = (df.iloc[source[key]].values > 0).astype(np.float32)
    return genes, mat


def export_clusters(data_dir: Path, out_dir: Path, K: int, seed: int, top_n: int):
    num_hyperedges = K * 2
    mirna_feat, drug_feat = load_features_and_concatenate(str(data_dir), "concatenate")
    mirna_df = pd.read_csv(data_dir / "miRNA.csv")
    drug_df = pd.read_csv(data_dir / "drug.csv")
    mirna_names = mirna_df["miRNAname"].tolist()
    drug_names = drug_df["drug"].tolist()
    n_m = len(mirna_names)

    features = np.concatenate([mirna_feat, drug_feat], axis=0)
    feat_norm = normalize(features, axis=1, norm="l2")
    clusters = KMeans(n_clusters=num_hyperedges, random_state=seed, n_init=10).fit_predict(feat_norm)

    rows = []
    for node_id, cid in enumerate(clusters):
        if node_id < n_m:
            rows.append(
                {"cluster_id": int(cid), "node_type": "miRNA", "node_id": int(node_id), "node_name": mirna_names[node_id]}
            )
        else:
            did = node_id - n_m
            rows.append(
                {"cluster_id": int(cid), "node_type": "drug", "node_id": int(node_id), "node_name": drug_names[did]}
            )
    membership = pd.DataFrame(rows)
    membership.to_csv(out_dir / "cluster_membership.csv", index=False, encoding="utf-8-sig")

    genes, mirna_gene = load_mirna_gene_union_maps(data_dir, mirna_names)
    summaries = []
    for cid in range(num_hyperedges):
        sub = membership[membership["cluster_id"] == cid]
        mirna_ids = sub[sub["node_type"] == "miRNA"]["node_id"].tolist()
        gene_mask = np.zeros(len(genes), dtype=bool)
        n_with_targets = 0
        for mid in mirna_ids:
            if mirna_gene[mid].sum() > 0:
                n_with_targets += 1
                gene_mask |= mirna_gene[mid] > 0
        summaries.append(
            {
                "cluster_id": cid,
                "n_mirna": int((sub["node_type"] == "miRNA").sum()),
                "n_drug": int((sub["node_type"] == "drug").sum()),
                "n_mirna_with_targets": n_with_targets,
                "n_genes_union": int(gene_mask.sum()),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["n_mirna_with_targets", "n_genes_union"], ascending=False
    )
    summary.to_csv(out_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig")
    top_ids = summary.head(top_n)["cluster_id"].tolist()
    return membership, summary, genes, mirna_gene, top_ids


def build_multihit_lists(membership, genes, mirna_gene, top_ids, out_dir: Path, min_hits: int):
    gene_dir = out_dir / "gene_lists_multihit"
    gene_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for cid in top_ids:
        mirna_ids = membership[
            (membership["cluster_id"] == cid) & (membership["node_type"] == "miRNA")
        ]["node_id"].tolist()
        if not mirna_ids:
            continue
        hit_counts = mirna_gene[mirna_ids].sum(axis=0)
        multi = [genes[j] for j, c in enumerate(hit_counts) if c >= min_hits]
        union = [genes[j] for j, c in enumerate(hit_counts) if c >= 1]
        use = multi if len(multi) >= 20 else union
        mode = f">={min_hits} hits" if use is multi and len(multi) >= 20 else "union_fallback"
        path = gene_dir / f"cluster_{cid}_genes.txt"
        path.write_text("\n".join(use), encoding="utf-8")
        records.append(
            {"cluster_id": cid, "n_genes_used": len(use), "n_union": len(union), "n_multihit": len(multi), "mode": mode, "gene_list_path": str(path)}
        )
        print(f"cluster {cid}: {len(use)} genes ({mode})")
    return pd.DataFrame(records)


def run_enrichr(gene_list, gene_sets, outdir: Path, prefix: str):
    import gseapy as gp

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism="human",
        outdir=None,
        cutoff=1.0,
        no_plot=True,
        verbose=False,
    )
    res = enr.results
    if res is None or len(res) == 0:
        return pd.DataFrame()
    res = res.copy()
    fdr_col = next((c for c in ("Adjusted P-value", "FDR", "P-value") if c in res.columns), None)
    if fdr_col:
        res = res.sort_values(fdr_col)
    res.to_csv(outdir / f"{prefix}_enrichr.csv", index=False, encoding="utf-8-sig")
    return res


def plot_kegg(sig: pd.DataFrame, out_png: Path, top_k: int = 8):
    if sig.empty:
        print("[WARN] no significant KEGG terms to plot")
        return
    fdr_col = "Adjusted P-value" if "Adjusted P-value" in sig.columns else "P-value"
    cids = sig.groupby("cluster_id").size().sort_values(ascending=False).head(3).index.tolist()
    plot_df = sig[sig["cluster_id"].isin(cids)]
    n = plot_df["cluster_id"].nunique()
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    for ax, (cid, sub) in zip(axes.ravel(), plot_df.groupby("cluster_id")):
        sub = sub.sort_values(fdr_col).head(top_k).iloc[::-1]
        ax.barh(sub["Term"].astype(str).str.slice(0, 55), -np.log10(sub[fdr_col].clip(lower=1e-300)))
        ax.axvline(-np.log10(0.05), color="gray", ls="--", lw=1)
        ax.set_title(f"Community {cid}")
        ax.set_xlabel(r"$-\log_{10}$(FDR)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Hypergraph community enrichment (KEGG)")
    ap.add_argument("--data_dir", type=Path, default=REPO / "data")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--K", type=int, default=25, help="causal prototypes; hyperedges = 2K")
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument("--min_hits", type=int, default=2)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_go", action="store_true", help="Only run KEGG")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    membership, summary, genes, mirna_gene, top_ids = export_clusters(
        args.data_dir, args.out, args.K, args.seed, args.top_n
    )
    print("[Top clusters]")
    print(summary.head(args.top_n).to_string(index=False))

    meta = build_multihit_lists(membership, genes, mirna_gene, top_ids, args.out, args.min_hits)
    libraries = ["KEGG_2021_Human"] if args.skip_go else ["KEGG_2021_Human", "GO_Biological_Process_2023"]
    outdir = args.out / "enrichr_results"
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for _, row in meta.iterrows():
        cid = int(row["cluster_id"])
        glist = [ln.strip() for ln in Path(row["gene_list_path"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
        for lib in libraries:
            prefix = f"cluster{cid}_{lib}"
            print(f"[Enrichr] cluster {cid} / {lib} (n={len(glist)}) ...")
            try:
                res = run_enrichr(glist, lib, outdir, prefix)
            except Exception as e:
                print(f"  FAILED: {e}")
                continue
            if res.empty:
                continue
            res["cluster_id"] = cid
            res["library"] = lib
            frames.append(res)

    if not frames:
        raise SystemExit("No enrichment results. Check gseapy install / network.")

    all_res = pd.concat(frames, ignore_index=True)
    fdr_col = "Adjusted P-value" if "Adjusted P-value" in all_res.columns else "P-value"
    sig = all_res[all_res[fdr_col] < args.fdr].copy()
    top = sig.sort_values(fdr_col).groupby(["cluster_id", "library"], as_index=False).head(10)
    top_path = args.out / "enrichment_top_clusters.csv"
    top.to_csv(top_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {top_path}")

    kegg_sig = sig[sig["library"].str.contains("KEGG", na=False)]
    plot_kegg(kegg_sig, args.out / "fig_enrichment_KEGG.png")
    print("\nDone. Outputs are under:", args.out)


if __name__ == "__main__":
    main()
