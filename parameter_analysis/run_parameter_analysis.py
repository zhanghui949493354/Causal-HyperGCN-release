import pandas as pd
import os
import sys
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from train import main_new  # noqa: E402

def run_parameter_analysis(target_cv=None):
    """
    Parameter sensitivity analysis.
    """
    output_dir = REPO / 'results' / 'parameter_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Shared base configuration
    base_config = {
        'fusion_method': 'multiply_residual',
        'gnn_type': 'causal_hyper',
        'predictor_type': 'xgboost',
        'negative_sampling_method': 'kmeans_fractal_cosine',
        'max_meta_path_length': 4,
        'gnn_layers': 4,
        'hidden_dim': 512,
        'num_prototypes': 25,
        'dropout': 0.4,
        'cv_type': 'pair',
        'num_epochs': 250,
        'seed': 42,
    }

    # Select the requested CV settings.
    if target_cv:
        cv_types = [target_cv]
        save_suffix = f'_{target_cv}'
    else:
        cv_types = ['pair', 'drug', 'mirna']
        save_suffix = ''

    print(f"Starting parameter sensitivity analysis (target: {cv_types})...")
    print(f"Base Config: {base_config}")

    # ==========================================================================
    # 1. Environment-prototype count.
    # ==========================================================================
    # Environment-prototype count.
    proto_options = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for cv in cv_types:
        for proto in proto_options:
            cfg = base_config.copy()
            cfg.update({
                'cv_type': cv,
                'num_prototypes': proto,
                'phase': '1_Prototypes'
            })
            cfg['desc'] = f"Proto_{proto}_{cv}"
            results.append(cfg)

    # ==========================================================================
    # 2. Hidden representation dimension.
    # ==========================================================================
    # Hidden representation dimension.
    dim_options = [32, 64, 128, 256, 512, 1024]

    for cv in cv_types:
        for dim in dim_options:
            cfg = base_config.copy()
            cfg['num_prototypes'] = 40 if cv == 'mirna' else 25
            cfg.update({
                'cv_type': cv,
                'hidden_dim': dim,
                'phase': '2_Dimension'
            })
            cfg['desc'] = f"Dim_{dim}_{cv}"
            results.append(cfg)

    # ==========================================================================
    # 3. Encoder depth.
    # ==========================================================================
    # Encoder depth.
    layer_options = [1, 2, 3, 4, 5]

    for cv in cv_types:
        for layers in layer_options:
            cfg = base_config.copy()
            cfg['num_prototypes'] = 40 if cv == 'mirna' else 25
            cfg.update({
                'cv_type': cv,
                'gnn_layers': layers,
                'phase': '3_Layers'
            })
            cfg['desc'] = f"Layer_{layers}_{cv}"
            results.append(cfg)

    # ==========================================================================
    # 4. Maximum meta-path length.
    # ==========================================================================
    # Maximum meta-path length.
    mp_options = [1, 2, 3, 4, 5, 6]

    for cv in cv_types:
        for mp in mp_options:
            cfg = base_config.copy()
            cfg['num_prototypes'] = 40 if cv == 'mirna' else 25
            cfg.update({
                'cv_type': cv,
                'max_meta_path_length': mp,
                'phase': '4_MetaPath'
            })
            cfg['desc'] = f"MP_{mp}_{cv}"
            results.append(cfg)

    # ==========================================================================
    # 5. Dropout rate.
    # ==========================================================================
    # Parameter search ranges
    dropout_options = [0.2, 0.3, 0.4, 0.5]

    for cv in cv_types:
        for drop in dropout_options:
            cfg = base_config.copy()
            cfg['num_prototypes'] = 40 if cv == 'mirna' else 25
            cfg.update({
                'cv_type': cv,
                'dropout': drop,
                'phase': '5_Dropout'
            })
            cfg['desc'] = f"Drop_{drop}_{cv}"
            results.append(cfg)

    # ==========================================================================
    # Run the planned experiments.
    # ==========================================================================
    total_exps = len(results)
    print(f"Total experiments planned: {total_exps}")

    final_records = []

    for i, cfg in enumerate(results):
        desc = cfg.pop('desc')
        phase = cfg.pop('phase')

        print(f"\n[{i+1}/{total_exps}] Running: {desc}")

        try:
            metrics = main_new(**cfg)

            rec = {'Experiment': desc, 'Phase': phase}
            rec.update(cfg)
            rec.update(metrics)
            final_records.append(rec)

            # Checkpoint completed records after every run.
            temp_df = pd.DataFrame(final_records)
            temp_path = os.path.join(output_dir, f'parameter_analysis_running{save_suffix}.csv')
            temp_df.to_csv(temp_path, index=False)

        except Exception as e:
            print(f" Error in {desc}: {e}")
            import traceback
            traceback.print_exc()

    final_df = pd.DataFrame(final_records)

    # Place the main metrics first.
    cols = ['Phase', 'Experiment', 'cv_type', 'auc', 'auprc', 'f1', 'precision', 'recall']
    other_cols = [c for c in final_df.columns if c not in cols]
    final_df = final_df[cols + other_cols]

    save_path = os.path.join(output_dir, f'parameter_analysis{save_suffix}.csv')
    final_df.to_csv(save_path, index=False)
    print(f"\n All completed. Results saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-type', dest='cv_type', type=str, choices=['pair', 'drug', 'mirna'], help='Specify CV type to run')
    args = parser.parse_args()

    run_parameter_analysis(target_cv=args.cv_type)
