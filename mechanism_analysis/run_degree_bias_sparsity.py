"""Generate the degree-bias and sparsity reports used in the mechanism analysis."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from train import main_new  # noqa: E402


def run_one(gnn_type, cv_type, epochs):
    return main_new(
        fusion_method="multiply_residual",
        gnn_type=gnn_type,
        predictor_type="mlp",
        cv_type=cv_type,
        negative_sampling_method="kmeans_fractal_cosine",
        gnn_layers=4,
        hidden_dim=512,
        dropout=0.4,
        num_prototypes=40 if cv_type == "mirna" else 25,
        max_meta_path_length=4,
        num_epochs=epochs,
        use_marginalization=True,
        enable_bias_sparsity_report=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", choices=["pair", "drug", "mirna"], required=True)
    parser.add_argument("--model", choices=["gcn", "causal_hyper"], default="causal_hyper")
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()
    run_one(args.model, args.cv, args.epochs)
