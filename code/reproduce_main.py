import os
import pandas as pd
from datetime import datetime
from train import main_new


def run_main_configs():
    configs = [
        {
            "name": "pair",
            "cv_type": "pair",
            "hidden_dim": 512,
            "gnn_layers": 4,
            "num_prototypes": 25,
            "max_meta_path_length": 4,
            "dropout": 0.4
        },
        {
            "name": "drug",
            "cv_type": "drug",
            "hidden_dim": 512,
            "gnn_layers": 4,
            "num_prototypes": 25,
            "max_meta_path_length": 4,
            "dropout": 0.4
        },
        {
            "name": "mirna",
            "cv_type": "mirna",
            "hidden_dim": 512,
            "gnn_layers": 4,
            "num_prototypes": 40,
            "max_meta_path_length": 4,
            "dropout": 0.4
        }
    ]

    common = {
        "fusion_method": "multiply_residual",
        "gnn_type": "causal_hyper",
        "predictor_type": "xgboost",
        "negative_sampling_method": "kmeans_fractal_cosine",
        "num_epochs": 250,
        "seed": 42
    }

    results = []
    out_dir = "./results/main"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for cfg in configs:
        params = {}
        params.update(common)
        params.update(cfg)
        name = params.pop("name")
        metrics = main_new(**params)
        record = {"mode": name}
        record.update(params)
        record.update(metrics)
        results.append(record)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f"main_cv_metrics_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to: {csv_path}")


if __name__ == "__main__":
    run_main_configs()
