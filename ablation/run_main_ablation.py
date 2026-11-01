"""Run representative component ablations."""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from train import main_new  # noqa: E402


BASE_CONFIG = {
    "fusion_method": "multiply_residual",
    "gnn_type": "causal_hyper",
    "predictor_type": "xgboost",
    "negative_sampling_method": "kmeans_fractal_cosine",
    "max_meta_path_length": 4,
    "gnn_layers": 4,
    "hidden_dim": 512,
    "num_prototypes": 25,
    "dropout": 0.4,
}

VARIANTS = {
    "full": {},
    "without_causal_prototypes": {"num_prototypes": 0},
    "without_marginalization": {"use_marginalization": False},
    "single_prototype": {"num_prototypes": 1},
    "local_gcn_only": {"gnn_type": "causal_gcn"},
    "global_hypergraph_only": {"gnn_type": "causal_hyper_global_only"},
    "standard_gcn": {"gnn_type": "gcn"},
    "metapath_length_1": {"max_meta_path_length": 1},
    "concatenation_fusion": {"fusion_method": "concatenate"},
    "mlp_predictor": {"predictor_type": "mlp"},
    "random_negative_sampling": {"negative_sampling_method": "random"},
    "kmeans_cosine_negative_sampling": {"negative_sampling_method": "kmeans_cosine"},
}


def run_ablation(cv_type, variants):
    output_dir = REPO / "results" / "ablation" / cv_type
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in variants:
        config = {**BASE_CONFIG, **VARIANTS[name], "cv_type": cv_type}
        if cv_type == "mirna" and "num_prototypes" not in VARIANTS[name]:
            config["num_prototypes"] = 40
        print(f"Running {name} under {cv_type} CV")
        metrics = main_new(**config)
        records.append({"variant": name, **config, **metrics})
        pd.DataFrame(records).to_csv(output_dir / "ablation_running.csv", index=False)

    result_path = output_dir / "ablation.csv"
    pd.DataFrame(records).to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", choices=["pair", "drug", "mirna"], required=True)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        help="Variant to run; repeat the option to run several. Defaults to all.",
    )
    args = parser.parse_args()
    run_ablation(args.cv, args.variant or list(VARIANTS))
