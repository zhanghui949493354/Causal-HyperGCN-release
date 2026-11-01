"""Deterministic, mutually exclusive cross-validation splits."""

import numpy as np


def build_fixed_cv_splits(positive_pairs, cv_type, num_folds=5, seed=42):
    """Create one seeded partition and reuse its disjoint folds.

    Pair-wise CV partitions association-row indices. Drug-wise and miRNA-wise
    CV partition the corresponding entity IDs. Every association/entity is in
    exactly one test fold, and all remaining units form that fold's training
    set.
    """
    pairs = np.asarray(positive_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("positive_pairs must have shape (n, 2)")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")

    rng = np.random.RandomState(seed)

    if cv_type == "pair":
        units = np.arange(len(pairs), dtype=np.int64)
        shuffled = rng.permutation(units)
        test_groups = np.array_split(shuffled, num_folds)
        splits = []
        for test_indices in test_groups:
            train_mask = np.ones(len(pairs), dtype=bool)
            train_mask[test_indices] = False
            splits.append((pairs[train_mask], pairs[test_indices], test_indices.copy()))
    elif cv_type in {"drug", "mirna"}:
        column = 1 if cv_type == "drug" else 0
        units = np.unique(pairs[:, column])
        shuffled = rng.permutation(units)
        test_groups = np.array_split(shuffled, num_folds)
        splits = []
        for test_entities in test_groups:
            test_mask = np.isin(pairs[:, column], test_entities)
            splits.append((pairs[~test_mask], pairs[test_mask], test_entities.copy()))
    else:
        raise ValueError(f"Unsupported fixed CV type: {cv_type}")

    test_units = np.concatenate([split[2] for split in splits])
    if len(test_units) != len(units) or len(np.unique(test_units)) != len(units):
        raise AssertionError("Test folds are not a disjoint, complete partition")

    return splits
