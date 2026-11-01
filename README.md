# Causal-HyperGCN

Causal-HyperGCN is a causality-enhanced multi-scale hypergraph convolutional network for miRNA--drug sensitivity association prediction.

## Repository structure

```text
code/                  Core model and cross-validation pipeline
data/                  Input data used by the pipeline
ablation/              Component-ablation entry point
parameter_analysis/    Parameter analysis and grid search
case_study/            Drug- or miRNA-oriented case study
external_validation/   Independent hold-out evaluation
enrichment/            Hypergraph-community enrichment analysis
mechanism_analysis/    Degree-bias and sparsity analysis
```

## Requirements

```bash
pip install -r requirements.txt
```

Recommended environment: Python 3.8.20, PyTorch 2.1.0, and CUDA 12.1.

## Quick start

Run Pair-wise, Drug-wise, and miRNA-wise cross-validation:

```bash
python code/reproduce_main.py
```

Run component ablation:

```bash
python ablation/run_main_ablation.py --cv pair
```

Run parameter analysis or grid search:

```bash
python parameter_analysis/run_parameter_analysis.py --cv-type pair
python parameter_analysis/run_grid_search.py --cv pair
```

Run a case study for a user-specified drug or miRNA:

```bash
python case_study/run_case_study.py --entity-type drug --target-name Cisplatin --top-k 14
```

Run external validation, enrichment, or mechanism analysis:

```bash
python external_validation/run_external_validation.py
python enrichment/run_enrichment.py
python mechanism_analysis/run_degree_bias_sparsity.py --cv drug --model causal_hyper
```

## Data

The `data/` directory contains association data, entity tables, node features, semantic features, and miRNA--gene and drug--gene matrices used by the code.

## Contact

- Hui Zhang: 2024388334@stu.zjhu.edu.cn
- Hailong Hu: 03139@zjhu.edu.cn
