"""Rank candidate associations for a user-selected drug or miRNA under a cold-start setting."""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from data_loader import load_biological_data_with_metapaths
from main import generate_negative_pairs
from gnn import ImprovedGNN
from metapath import ImprovedMetaPathEncoder
from predictor import HybridPredictor
from models import AttentionFusion, LabelSmoothingBCE
from train import Config, EarlyFusionProjector


class CaseStudyConfig(Config):
    """Default model settings for case-study analysis."""

    def __init__(self, num_prototypes):
        super().__init__()
        self.fusion_method = "multiply_residual"
        self.gnn_type = "causal_hyper"
        self.use_virtual_node = False
        self.predictor_type = "xgboost"
        self.negative_sampling_method = "kmeans_fractal_cosine"
        self.output_dim = 512
        self.gnn_layers = 4
        self.num_prototypes = num_prototypes
        self.dropout = 0.4
        self.max_meta_path_length = 4
        self.seed = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def renormalize_rows(matrix):
    matrix = matrix.copy()
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def prepare_target_holdout(
    positive_pairs, adj_matrix, meta_paths, target_node_id, target_kind
):
    """Remove every known association involving one target before training."""
    pair_column = 1 if target_kind == "drug" else 0
    holdout_mask = positive_pairs[:, pair_column] == target_node_id
    held_out_pairs = positive_pairs[holdout_mask].copy()
    train_positive_pairs = positive_pairs[~holdout_mask].copy()

    masked_adj = adj_matrix.copy()
    masked_adj[target_node_id, :] = 0.0
    masked_adj[:, target_node_id] = 0.0

    masked_meta_paths = {name: matrix.copy() for name, matrix in meta_paths.items()}
    if "mirna_drug" in masked_meta_paths:
        association_path = masked_meta_paths["mirna_drug"]
        association_path[target_node_id, :] = 0.0
        association_path[:, target_node_id] = 0.0
        masked_meta_paths["mirna_drug"] = renormalize_rows(association_path)

    if len(held_out_pairs) == 0:
        raise ValueError(
            f"The target node {target_node_id} has no known associations to hold out."
        )
    return train_positive_pairs, held_out_pairs, masked_adj, masked_meta_paths


def generate_training_pairs(
    train_positive_pairs, features, masked_adj, excluded_node_id, config
):
    """Generate negatives without allowing the held-out entity into training."""
    eligible_nodes = np.delete(np.arange(len(features)), excluded_node_id)
    global_to_local = {node_id: idx for idx, node_id in enumerate(eligible_nodes)}
    local_positive_pairs = np.asarray(
        [[global_to_local[u], global_to_local[v]] for u, v in train_positive_pairs],
        dtype=np.int64,
    )

    result = generate_negative_pairs(
        local_positive_pairs,
        len(eligible_nodes),
        len(local_positive_pairs),
        method=config.negative_sampling_method,
        features=features[eligible_nodes],
        adj_matrix=masked_adj[np.ix_(eligible_nodes, eligible_nodes)],
    )
    local_negative_pairs = result[0] if isinstance(result, tuple) else result
    train_negative_pairs = eligible_nodes[np.asarray(local_negative_pairs, dtype=np.int64)]

    train_pairs = np.vstack([train_positive_pairs, train_negative_pairs])
    train_labels = np.concatenate(
        [np.ones(len(train_positive_pairs)), np.zeros(len(train_negative_pairs))]
    ).astype(np.float32)
    if np.any(train_pairs == excluded_node_id):
        raise RuntimeError("The held-out target was included in a training pair.")
    order = np.random.permutation(len(train_pairs))
    return train_pairs[order], train_labels[order]


def build_hypergraph(features, config, device, excluded_node_id):
    num_hyperedges = config.num_prototypes * 2
    train_node_ids = np.delete(np.arange(len(features)), excluded_node_id)
    normalized_features = normalize(np.asarray(features), axis=1, norm="l2")
    num_hyperedges = max(1, min(int(num_hyperedges), len(train_node_ids)))
    model = KMeans(
        n_clusters=num_hyperedges,
        random_state=config.seed,
        n_init=10,
    )
    model.fit(normalized_features[train_node_ids])
    labels = model.predict(normalized_features)
    incidence = np.zeros((len(normalized_features), num_hyperedges), dtype=np.float32)
    incidence[np.arange(len(normalized_features)), labels] = 1.0
    return torch.as_tensor(incidence, dtype=torch.float32, device=device)


def train_target_model(
    features, masked_adj, masked_meta_paths, train_positive_pairs, target_node_id,
    config, device
):
    train_pairs, train_labels = generate_training_pairs(
        train_positive_pairs, features, masked_adj, target_node_id, config
    )

    projector = EarlyFusionProjector(
        fusion_method=config.fusion_method,
        output_dim=config.output_dim,
        device=device,
    ).to(device)
    gnn = ImprovedGNN(
        input_dim=config.output_dim,
        hidden_dims=[config.output_dim] * config.gnn_layers,
        dropout=config.dropout,
        gnn_type=config.gnn_type,
        use_virtual_node=config.use_virtual_node,
        num_prototypes=config.num_prototypes,
    ).to(device)
    metapath_encoder = ImprovedMetaPathEncoder(
        input_dim=config.output_dim,
        hidden_dim=config.output_dim,
        dropout=config.dropout,
        use_attention=True,
    ).to(device)
    fusion = AttentionFusion(feat_dim=config.output_dim).to(device)
    predictor = HybridPredictor(
        predictor_type=config.predictor_type,
        input_dim=config.output_dim * 2,
        device=device,
    )

    optimizer = optim.AdamW(
        list(projector.parameters())
        + list(gnn.parameters())
        + list(metapath_encoder.parameters())
        + list(fusion.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = LabelSmoothingBCE(alpha=0.1)

    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    adj_tensor = torch.as_tensor(masked_adj, dtype=torch.float32, device=device)
    meta_tensors = {
        name: torch.as_tensor(matrix, dtype=torch.float32, device=device)
        for name, matrix in masked_meta_paths.items()
    }
    label_tensor = torch.as_tensor(train_labels, dtype=torch.float32, device=device)
    incidence = build_hypergraph(features, config, device, target_node_id)

    for epoch in range(config.num_epochs):
        projector.train()
        gnn.train()
        metapath_encoder.train()
        fusion.train()
        optimizer.zero_grad()

        shared = projector(feature_tensor)
        graph_embeddings, posterior = gnn(
            shared, adj_tensor, H=incidence, causal_inference=False
        )
        path_embeddings = metapath_encoder(shared, meta_tensors)
        fused_embeddings = fusion([graph_embeddings, path_embeddings])
        if fused_embeddings.shape == shared.shape:
            fused_embeddings = fused_embeddings + shared

        first = fused_embeddings[train_pairs[:, 0]]
        second = fused_embeddings[train_pairs[:, 1]]
        proxy_predictions = torch.sigmoid((first * second).sum(dim=1))
        loss = criterion(proxy_predictions, label_tensor)
        if posterior is not None:
            average_posterior = posterior.mean(dim=0)
            entropy = -torch.sum(
                average_posterior * torch.log(average_posterior + 1e-8)
            )
            loss = loss - 0.01 * entropy
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch + 1 == config.num_epochs:
            print(
                f"    epoch {epoch + 1:>3}/{config.num_epochs}: "
                f"loss={loss.item():.4f}"
            )

    projector.eval()
    gnn.eval()
    metapath_encoder.eval()
    fusion.eval()
    with torch.no_grad():
        shared = projector(feature_tensor)
        graph_embeddings, _ = gnn(
            shared, adj_tensor, H=incidence, causal_inference=True
        )
        path_embeddings = metapath_encoder(shared, meta_tensors)
        final_embeddings = fusion([graph_embeddings, path_embeddings])
        if final_embeddings.shape == shared.shape:
            final_embeddings = final_embeddings + shared

    first = final_embeddings[train_pairs[:, 0]]
    second = final_embeddings[train_pairs[:, 1]]
    predictor.fit_xgboost(torch.cat([first, second], dim=1), label_tensor)
    return final_embeddings, predictor


def score_candidates(candidate_pairs, embeddings, predictor, batch_size=4096):
    scores = []
    with torch.no_grad():
        for start in range(0, len(candidate_pairs), batch_size):
            batch = candidate_pairs[start : start + batch_size]
            first = embeddings[batch[:, 0]]
            second = embeddings[batch[:, 1]]
            pair_embeddings = torch.cat([first, second], dim=1)
            scores.extend(predictor.predict_xgboost(pair_embeddings).cpu().numpy())
    return np.asarray(scores)


def run_target(
    target_name, target_kind, features, adj_matrix, positive_pairs, meta_paths,
    drug_df, mirna_df, drug_name_to_id, mirna_name_to_id, output_dir,
    top_k, epochs, device
):
    os.makedirs(output_dir, exist_ok=True)
    mirna_count = len(mirna_df)
    drug_count = len(drug_df)
    if target_kind == "drug":
        if target_name not in drug_name_to_id:
            raise KeyError(f"Drug not found in the dataset: {target_name}")
        target_node_id = int(drug_name_to_id[target_name]) + mirna_count
        candidate_pairs = np.column_stack(
            [np.arange(mirna_count), np.full(mirna_count, target_node_id)]
        )
        candidate_names = dict(zip(mirna_df["id"], mirna_df["miRNAname"]))
        candidate_ids = candidate_pairs[:, 0]
        num_prototypes = 25
        name_column = "miRNA"
    else:
        if target_name not in mirna_name_to_id:
            raise KeyError(f"miRNA not found in the dataset: {target_name}")
        target_node_id = int(mirna_name_to_id[target_name])
        drug_node_ids = np.arange(drug_count) + mirna_count
        candidate_pairs = np.column_stack(
            [np.full(drug_count, target_node_id), drug_node_ids]
        )
        candidate_names = dict(zip(drug_df["id"], drug_df["drug"]))
        candidate_ids = candidate_pairs[:, 1] - mirna_count
        num_prototypes = 40
        name_column = "Drug"

    config = CaseStudyConfig(num_prototypes=num_prototypes)
    config.num_epochs = epochs
    set_seed(config.seed)
    train_positives, held_out, masked_adj, masked_meta = prepare_target_holdout(
        positive_pairs, adj_matrix, meta_paths, target_node_id, target_kind
    )
    print(
        f"\n[{target_kind}] {target_name}: held out {len(held_out)} known "
        f"associations; training with {len(train_positives)} positives; K={num_prototypes}"
    )
    embeddings, predictor = train_target_model(
        features, masked_adj, masked_meta, train_positives, target_node_id,
        config, device
    )

    known_pairs = set(map(tuple, positive_pairs.tolist()))
    unknown_mask = np.array(
        [tuple(pair) not in known_pairs for pair in candidate_pairs], dtype=bool
    )
    unknown_pairs = candidate_pairs[unknown_mask]
    unknown_ids = candidate_ids[unknown_mask]
    scores = score_candidates(unknown_pairs, embeddings, predictor)
    order = np.argsort(-scores)[:top_k]
    result = pd.DataFrame(
        {
            "Rank": np.arange(1, len(order) + 1),
            name_column: [candidate_names[int(unknown_ids[i])] for i in order],
            "Score": scores[order],
        }
    )
    safe_name = target_name.replace("/", "_").replace(" ", "_")
    output_path = os.path.join(
        output_dir, f"{target_kind}_{safe_name}_top{top_k}.csv"
    )
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False))
    print(f"    saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Target-specific cold-start case-study reproduction."
    )
    parser.add_argument("--entity-type", choices=("drug", "mirna"), required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./case_study_results")
    parser.add_argument("--epochs", type=int, default=250)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    base_config = CaseStudyConfig(num_prototypes=25)
    (
        features,
        adj_matrix,
        positive_pairs,
        meta_paths,
        drug_df,
        mirna_df,
        drug_name_to_id,
        mirna_name_to_id,
        _,
    ) = load_biological_data_with_metapaths(
        args.data_dir, "concatenate", base_config.max_meta_path_length
    )

    run_target(
        args.target_name, args.entity_type, features, adj_matrix, positive_pairs,
        meta_paths, drug_df, mirna_df, drug_name_to_id, mirna_name_to_id,
        args.output_dir, args.top_k, args.epochs, device
    )


if __name__ == "__main__":
    main()
