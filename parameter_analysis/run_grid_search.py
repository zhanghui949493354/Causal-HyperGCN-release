"""Grid search entry point for one of the three cross-validation protocols."""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from train import main_new  # noqa: E402


def run_grid_search(cv_type):
    output_dir = REPO / "results" / "grid_search" / cv_type
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_grid = {
        "hidden_dim": [32, 64, 128, 256, 512, 1024],
        "gnn_layers": [1, 2, 3, 4, 5],
        "num_prototypes": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "dropout": [0.2, 0.3, 0.4, 0.5],
        "max_meta_path_length": [1, 2, 3, 4, 5, 6],
    }
    base_config = {
        "fusion_method": "multiply_residual",
        "gnn_type": "causal_hyper",
        "predictor_type": "xgboost",
        "negative_sampling_method": "kmeans_fractal_cosine",
        "cv_type": cv_type,
    }

    keys = list(parameter_grid)
    records = []
    combinations = itertools.product(*(parameter_grid[key] for key in keys))
    for index, values in enumerate(combinations, start=1):
        varied = dict(zip(keys, values))
        config = {**base_config, **varied}
        description = "_".join([cv_type] + [f"{key}={value}" for key, value in varied.items()])
        print(f"[{index}] {description}")
        metrics = main_new(**config)
        records.append({"experiment": description, **config, **metrics})
        pd.DataFrame(records).to_csv(output_dir / "grid_search_running.csv", index=False)

    result_path = output_dir / "grid_search.csv"
    pd.DataFrame(records).to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", choices=["pair", "drug", "mirna"], required=True)
    args = parser.parse_args()
    run_grid_search(args.cv)
