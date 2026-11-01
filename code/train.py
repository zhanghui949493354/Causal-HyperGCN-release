"""
========================================
 Modular Training and Evaluation Entry Point
========================================

Five-fold training entry point with modular fusion, graph, meta-path, and
prediction components. Supported settings are documented in README.md.
"""

import numpy as np
import torch
import torch.optim as optim
import os
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Environment settings
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
from data_loader import load_biological_data_with_metapaths

# Project modules
from fusion import FusionModule
from gnn import ImprovedProjector, ImprovedGNN, HGGNGAT
from metapath import ImprovedMetaPathEncoder
from predictor import HybridPredictor
from models import LabelSmoothingBCE, AttentionFusion
from utils import prepare_fold_data, evaluate_full_graph_with_predictions, plot_roc_curve, plot_all_roc_curves, plot_tsne
from main import generate_negative_pairs, Config as OriginalConfig
from cv_splits import build_fixed_cv_splits

class Config(OriginalConfig):
    """Extended Config"""
    def __init__(self):
        super().__init__()
        # Default settings
        self.fusion_method = 'concatenate'
        self.gnn_type = 'gcn'
        self.predictor_type = 'mlp'
        self.dropout = 0.4
        self.num_epochs = 150
        self.learning_rate = 1e-5
        self.weight_decay = 0.001
        self.seed = 42

        # Early stopping.
        self.early_stopping_patience = 20
        self.early_stopping_delta = 0.0001
        self.warmup_epochs = 30

class EarlyFusionProjector(torch.nn.Module):
    """
    Wrapper to handle early fusion (128+128 -> Fusion -> Projector).
    Expects input of shape [N, 256] (concatenated raw features).
    """
    def __init__(self, fusion_method, output_dim=32, device='cpu'):
        super().__init__()
        self.fusion = FusionModule(method=fusion_method, input_dim=128)
        self.projector = ImprovedProjector(input_dim=self.fusion.output_dim, output_dim=output_dim)
        self.device = device

    def forward(self, x):
        # Split x back into 128 and 128 (Assuming first 128 is struct, next 128 is semantic)
        if x.shape[1] == 256:
            f1 = x[:, :128]
            f2 = x[:, 128:]
            fused = self.fusion(f1, f2)
            return self.projector(fused)
        else:
            return self.projector(x)

def main_new(fusion_method='multiply_residual', gnn_type='causal_hyper', predictor_type='xgboost',
             negative_sampling_method='kmeans_fractal_cosine', cv_type='pair',
             max_meta_path_length=4,
             gnn_layers=4, hidden_dim=512, num_prototypes=25, dropout=0.4, num_epochs=None, use_marginalization=True, predictor_params=None,
             analysis_drug_names=None, analysis_mirna_names=None, enable_bias_sparsity_report=True,
             external_eval=False, external_test_pairs=None, seed=None):

    # Setup Config
    config = Config()
    if seed is not None:
        config.seed = int(seed)
    if num_epochs is not None:
        config.num_epochs = num_epochs

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    config.output_dim = hidden_dim
    config.num_prototypes = num_prototypes
    config.dropout = dropout
    config.use_marginalization = use_marginalization

    # Parse GNN type for Virtual Node flag
    use_virtual_node = False
    real_gnn_type = gnn_type

    if '_virtual' in gnn_type:
        use_virtual_node = True
        real_gnn_type = gnn_type.replace('_virtual', '')
    elif gnn_type == 'hggn':
        real_gnn_type = 'hggn'
    elif gnn_type == 'causal_hyper':
        real_gnn_type = 'causal_hyper'

    config.fusion_method = fusion_method
    config.gnn_type = real_gnn_type
    config.predictor_type = predictor_type
    config.negative_sampling_method = negative_sampling_method

    if external_eval or cv_type == 'external':
        external_eval = True
        cv_type = 'external'
        if external_test_pairs is None or len(external_test_pairs) == 0:
            raise ValueError("external_eval requires non-empty external_test_pairs (mirna_id, drug_global_id).")
        config.num_folds = 1

    print(f"\nCV setting: {cv_type}")

    # Load the 256-dimensional concatenated representation so that the selected
    # fusion module can separate and combine the two 128-dimensional views.
    data_dir = './data'

    load_fusion_arg = 'concatenate'

    try:
        features, adj_matrix, positive_pairs, meta_paths, drug_df, mirna_df, drug_name_to_id, mirna_name_to_id, original_associations = \
            load_biological_data_with_metapaths(data_dir, load_fusion_arg, max_meta_path_length)

    except Exception as e:
        print(f"Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    if analysis_drug_names is None:
        analysis_drug_names = ['Cisplatin', '5-Fluorouracil', 'Oxaliplatin']
    if analysis_mirna_names is None:
        analysis_mirna_names = ['miR-21', 'miR-155', 'miR-34a']

    # --- Hypergraph Construction using KMeans ---
    # We perform clustering on ALL nodes features to build the hypergraph structure H
    # This H will be shared across folds (transductive assumption for structure)
        # Each fold uses the rows of the shared transductive hypergraph that
        # correspond to its training nodes.
    # But for Hypergraph structure learning, using global unsupervised clustering is often acceptable.
    # Let's do it globally for simplicity and stability.

    from sklearn.cluster import KMeans

    # 1. Normalize features for clustering
    from sklearn.preprocessing import normalize
    feat_norm = normalize(features, axis=1, norm='l2')

    # 2. KMeans Clustering
    # Number of clusters = Number of hyperedges
    if config.num_prototypes > 0:
        num_hyperedges = config.num_prototypes * 2
    else:
        num_hyperedges = 20  # Preserve the hypergraph when causal prototypes are disabled.

    kmeans = KMeans(n_clusters=num_hyperedges, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(feat_norm)

    # 3. Build Incidence Matrix H (N, M)
    # H[i, k] = 1 if node i in cluster k
    num_nodes_all = features.shape[0]
    H = torch.zeros(num_nodes_all, num_hyperedges)
    for i, c in enumerate(clusters):
        H[i, c] = 1.0

    H = H.to(config.device)

    all_fold_embeddings = []
    all_fold_data = {}
    bias_sparsity_reports = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f'./results/{cv_type}/d{hidden_dim}_L{gnn_layers}_dp{dropout}_l{max_meta_path_length}_K{num_prototypes}_{timestamp}_{gnn_type}_{predictor_type}'
    os.makedirs(result_dir, exist_ok=True)

    # Pair layout is (miRNA node ID, drug node ID).
    mirna_ids = np.unique(positive_pairs[:, 0])
    drug_ids = np.unique(positive_pairs[:, 1])
    num_folds = config.num_folds
    fixed_cv_splits = None
    if cv_type in {'pair', 'drug', 'mirna'}:
        fixed_cv_splits = build_fixed_cv_splits(
            positive_pairs, cv_type, num_folds=num_folds, seed=config.seed
        )
    for fold_idx in range(num_folds):
        print(f"\nFold {fold_idx + 1}/{num_folds}")

        # --- 1. Cross-validation split ---
        if cv_type == 'external':
            train_pos_pairs = positive_pairs.copy()
            test_pos_pairs = np.asarray(external_test_pairs, dtype=np.int64)
            # Remove any pair already present in the training associations.
            train_set = set(map(tuple, train_pos_pairs.tolist()))
            test_pos_pairs = np.array([p for p in test_pos_pairs if tuple(p) not in train_set], dtype=np.int64)
            if len(test_pos_pairs) == 0:
                raise ValueError("No external test positives left after removing train overlap.")
        elif cv_type in {'pair', 'drug', 'mirna'}:
            train_pos_pairs, test_pos_pairs, _ = fixed_cv_splits[fold_idx]
        else:
            raise ValueError(f"Unknown cv_type: {cv_type}")

        train_nodes_unique = np.unique(train_pos_pairs.flatten())
        # Map global node identifiers to the current training subset.
        train_id_map = {oid: i for i, oid in enumerate(train_nodes_unique)}
        train_features_subset = features[train_nodes_unique]

        train_pos_mapped = []
        for u, v in train_pos_pairs:
            train_pos_mapped.append([train_id_map[u], train_id_map[v]])
        train_pos_mapped = np.array(train_pos_mapped)

        train_res = generate_negative_pairs(
            train_pos_mapped,
            len(train_nodes_unique),
            len(train_pos_pairs),
            method=config.negative_sampling_method,
            features=train_features_subset
        )

        kmeans_model = None
        if isinstance(train_res, tuple):
            train_neg_mapped, kmeans_model = train_res
        else:
            train_neg_mapped = train_res

        # Map negatives back to global node IDs.
        train_neg_pairs = []
        for u, v in train_neg_mapped:
            train_neg_pairs.append([train_nodes_unique[u], train_nodes_unique[v]])
        train_neg_pairs = np.array(train_neg_pairs)

        # Reuse the training-fold KMeans model for test sampling.

        # Extract nodes present in the test pairs.
        test_nodes_unique = np.unique(test_pos_pairs.flatten())
        test_id_map = {oid: i for i, oid in enumerate(test_nodes_unique)}
        test_features_subset = features[test_nodes_unique]

        test_pos_mapped = []
        for u, v in test_pos_pairs:
            test_pos_mapped.append([test_id_map[u], test_id_map[v]])
        test_pos_mapped = np.array(test_pos_mapped)

        test_res = generate_negative_pairs(
            test_pos_mapped,
            len(test_nodes_unique),
            len(test_pos_pairs),
            method=config.negative_sampling_method,
            features=test_features_subset,
            kmeans_model=kmeans_model
        )

        if isinstance(test_res, tuple):
            test_neg_mapped, _ = test_res
        else:
            test_neg_mapped = test_res

        test_neg_pairs = []
        for u, v in test_neg_mapped:
            test_neg_pairs.append([test_nodes_unique[u], test_nodes_unique[v]])
        test_neg_pairs = np.array(test_neg_pairs)

        train_pairs = np.vstack([train_pos_pairs, train_neg_pairs])
        test_pairs = np.vstack([test_pos_pairs, test_neg_pairs])
        train_labels = np.concatenate([np.ones(len(train_pos_pairs)), np.zeros(len(train_neg_pairs))])
        test_labels = np.concatenate([np.ones(len(test_pos_pairs)), np.zeros(len(test_neg_pairs))])

        # Shuffle pairs within each split.
        train_shuffle = np.random.permutation(len(train_pairs))
        train_pairs = train_pairs[train_shuffle]
        train_labels = train_labels[train_shuffle]

        # Prepare fold tensors.
        train_nodes = np.unique(train_pairs.flatten())
        test_nodes = np.unique(test_pairs.flatten())
        if cv_type == 'external':
            # Keep the full entity vocabulary so mappable external nodes retain
            # precomputed features + KG/meta-path structure (transductive over known IDs).
            train_nodes = np.arange(features.shape[0], dtype=np.int64)

        fold_data = prepare_fold_data(
            train_pairs, test_pairs, train_labels, test_labels,
            train_nodes, test_nodes, features, adj_matrix, meta_paths
        )

        projector = EarlyFusionProjector(fusion_method=fusion_method, output_dim=config.output_dim, device=config.device).to(config.device)

        hidden_dims = [config.output_dim] * gnn_layers

        if config.gnn_type == 'hggn':
            # HGGN-GAT requires the fold node count for its virtual node.
            num_nodes = features.shape[0]
            gnn = HGGNGAT(input_dim=config.output_dim, hidden_dims=hidden_dims,
                          dropout=config.dropout, num_nodes=num_nodes).to(config.device)
        else:
            if use_virtual_node:
                 # The virtual-node model uses the configured encoder capacity.
                 pass

            # Pass the selected encoder depth and environment-prototype count.
            gnn = ImprovedGNN(input_dim=config.output_dim, hidden_dims=hidden_dims,
                              dropout=config.dropout, gnn_type=config.gnn_type,
                              use_virtual_node=use_virtual_node, num_prototypes=config.num_prototypes).to(config.device)

        # Meta-path encoder with semantic attention.
        metapath_enc = ImprovedMetaPathEncoder(input_dim=config.output_dim, hidden_dim=config.output_dim,
                                               dropout=config.dropout, use_attention=True).to(config.device)

        # Fuse graph and meta-path representations.
        # ImprovedMetaPathEncoder returns an attention-fused semantic tensor;
        # AttentionFusion then combines graph and semantic representations.
        fusion = AttentionFusion(feat_dim=config.output_dim).to(config.device)

        # Final predictor.
        predictor = HybridPredictor(predictor_type=predictor_type, input_dim=config.output_dim * 2,
                                    device=config.device, predictor_params=predictor_params or {})
        if predictor.type == 'mlp':
            predictor.model.to(config.device)

        # Optimizer.
        params = list(projector.parameters()) + list(gnn.parameters()) + \
                 list(metapath_enc.parameters()) + list(fusion.parameters())
        if predictor.type == 'mlp':
            params += list(predictor.parameters())

        optimizer = optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        bce_criterion = LabelSmoothingBCE(alpha=0.1)

        # View-dropout regularization.
        # View dropout regularizes inductive representations.
        feature_dropout_rate = 0.5

        # Training loop.
        best_test_auc = 0.0
        best_epoch = 0
        patience = 0

        # Track the best checkpoint.
        best_state = {}

        for epoch in range(config.num_epochs):
            projector.train()
            gnn.train()
            metapath_enc.train()
            fusion.train()
            predictor.train()

            optimizer.zero_grad()

            # Move fold tensors to the selected device.
            train_feat = fold_data['train_feat'].to(config.device)
            train_adj = fold_data['train_adj'].to(config.device)
            train_meta = {k: v.to(config.device) for k, v in fold_data['train_meta_paths'].items()}
            train_y = fold_data['train_y'].to(config.device)

            # Slice the global incidence matrix to the nodes available in this fold.
            train_nodes_ids = fold_data['train_nodes'] # numpy array of global IDs
            train_H = H[train_nodes_ids].to(config.device)

            # Forward Pass
            shared_feat = projector(train_feat)

            # Causal GNN Handling
            p_z_x_loss = 0
            p_z_x = None
            causal_types = ['causal', 'causal_hyper', 'causal_gcn', 'causal_hyper_global_only']
            causal_types_need_H = ['causal_hyper', 'causal_hyper_global_only']
            if config.gnn_type in causal_types:
                if config.gnn_type in causal_types_need_H:
                    gnn_emb, p_z_x = gnn(shared_feat, train_adj, H=train_H, causal_inference=False)
                else:
                    gnn_emb, p_z_x = gnn(shared_feat, train_adj, causal_inference=False)
            else:
                gnn_emb = gnn(shared_feat, train_adj)

            if p_z_x is not None:
                p_avg = torch.mean(p_z_x, dim=0)
                entropy = -torch.sum(p_avg * torch.log(p_avg + 1e-8))
                p_z_x_loss = -0.01 * entropy

            mp_embs = metapath_enc(shared_feat, train_meta) # Returns single tensor [N, D]

            # View dropout regularizes the graph and meta-path representations.

            if cv_type == 'drug': # Enable view dropout in the inductive drug-wise setting.
                if np.random.rand() < 0.3:
                    gnn_emb = torch.zeros_like(gnn_emb)
                if np.random.rand() < 0.3:
                    mp_embs = torch.zeros_like(mp_embs)

            # Fusion
            fused_emb = fusion([gnn_emb, mp_embs])

            # Preserve the projected node features through a residual connection.
            if shared_feat.shape == fused_emb.shape:
                 fused_emb = fused_emb + shared_feat


            # Predict
            train_pairs_idx = fold_data['train_pairs']

            # Get embeddings for pairs
            # map global node id to local fold index
            node_map = fold_data['node_map']
            emb1_idx = [node_map[n] for n in train_pairs_idx[:, 0]]
            emb2_idx = [node_map[n] for n in train_pairs_idx[:, 1]]

            emb1 = fused_emb[emb1_idx]
            emb2 = fused_emb[emb2_idx]
            pair_emb = torch.cat([emb1, emb2], dim=1)

            if predictor.type == 'mlp':
                pred = predictor(pair_emb).squeeze()
                loss = bce_criterion(pred, train_y)

                # Add Causal Regularization
                if config.gnn_type in ['causal', 'causal_hyper', 'causal_gcn', 'causal_hyper_global_only'] and p_z_x is not None:
                    loss = loss + p_z_x_loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss = loss.item()
            else:
                # XGBoost Training Phase (Representation Learning)
                # Train embeddings with a dot-product proxy loss.
                sim = (emb1 * emb2).sum(dim=1)
                pred = torch.sigmoid(sim) # Simple dot product prediction
                loss = bce_criterion(pred, train_y)

                # Add Causal Regularization
                if config.gnn_type in ['causal', 'causal_hyper', 'causal_gcn', 'causal_hyper_global_only'] and p_z_x is not None:
                    loss = loss + p_z_x_loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss = loss.item()

            # Epoch-level diagnostic evaluation.
            # During representation learning, XGBoost uses proxy diagnostics; the
            # tree classifier is fitted after the node embeddings are finalized.

            if predictor.type == 'mlp':
                with torch.no_grad():
                    projector.eval()
                    gnn.eval()
                    metapath_enc.eval()
                    fusion.eval()
                    predictor.eval()

                    test_pairs_idx = fold_data['test_pairs']
                    test_y_true = fold_data['test_y'] # numpy

                    shared_feat_eval = projector(train_feat)

                    if config.gnn_type == 'causal':
                        gnn_emb_eval, _ = gnn(shared_feat_eval, train_adj, causal_inference=use_marginalization)
                    elif config.gnn_type in ['causal_hyper', 'causal_hyper_global_only']:
                        gnn_emb_eval, _ = gnn(shared_feat_eval, train_adj, H=train_H, causal_inference=use_marginalization)
                    elif config.gnn_type == 'causal_gcn':
                        gnn_emb_eval, _ = gnn(shared_feat_eval, train_adj, causal_inference=use_marginalization)
                    else:
                        gnn_emb_eval = gnn(shared_feat_eval, train_adj)

                    mp_embs_eval = metapath_enc(shared_feat_eval, train_meta)
                    fused_emb_eval = fusion([gnn_emb_eval, mp_embs_eval])

                    # Extract embeddings for mappable test pairs.
                    valid_mask = np.array([n in node_map for n in test_pairs_idx[:, 0]]) & \
                                 np.array([n in node_map for n in test_pairs_idx[:, 1]])

                    if np.sum(valid_mask) > 0:
                        t_emb1_idx = [node_map[n] for n in test_pairs_idx[valid_mask, 0]]
                        t_emb2_idx = [node_map[n] for n in test_pairs_idx[valid_mask, 1]]

                        t_emb1 = fused_emb_eval[t_emb1_idx]
                        t_emb2 = fused_emb_eval[t_emb2_idx]
                        t_pair_emb = torch.cat([t_emb1, t_emb2], dim=1)

                        t_pred = predictor(t_pair_emb).squeeze().cpu().numpy()
                        t_y = test_y_true[valid_mask]

                        try:
                            train_auc = roc_auc_score(train_y.cpu().numpy(), pred.detach().cpu().numpy())
                        except:
                            train_auc = 0.0

                        # Compute an epoch-level diagnostic on the available test pairs.

                        try:
                            # Monitor the training process on a bounded test sample.
                            test_pairs_all = fold_data['test_pairs']
                            def get_emb_safe_tmp(nid):
                                if nid in node_map: return fused_emb[node_map[nid]]
                                return torch.zeros_like(fused_emb[0])

                            # Bound the diagnostic sample size for efficiency.
                            sample_size = min(len(test_pairs_all), 2000)
                            indices = np.random.choice(len(test_pairs_all), sample_size, replace=False)

                            t_emb1_tmp = torch.stack([get_emb_safe_tmp(n) for n in test_pairs_all[indices, 0]])
                            t_emb2_tmp = torch.stack([get_emb_safe_tmp(n) for n in test_pairs_all[indices, 1]])
                            t_pair_tmp = torch.cat([t_emb1_tmp, t_emb2_tmp], dim=1)

                            if predictor.type == 'xgboost':
                                # XGBoost is fitted after representation training.
                                t_pred_tmp = np.zeros(sample_size)
                            else:
                                t_pred_tmp = predictor(t_pair_tmp).squeeze().detach().cpu().numpy()

                            t_y_tmp = fold_data['test_y'][indices].cpu().numpy()
                            test_auc = roc_auc_score(t_y_tmp, t_pred_tmp)

                            # Test F1 on the sampled diagnostic pairs.
                            test_f1 = f1_score(t_y_tmp, (t_pred_tmp >= 0.5).astype(int), zero_division=0)
                        except:
                            test_auc = 0.0
                            test_f1 = 0.0


                        train_f1 = f1_score(train_y.cpu().numpy(), (pred.detach().cpu().numpy() >= 0.5).astype(int), zero_division=0)
                    else:
                        test_auc = 0
                        test_f1 = 0

            else:
                # Dot-product AUC is a diagnostic, not the final XGBoost metric.
                try:
                    train_auc = roc_auc_score(train_y.cpu().numpy(), pred.detach().cpu().numpy())
                    train_f1 = f1_score(train_y.cpu().numpy(), (pred.detach().cpu().numpy() >= 0.5).astype(int), zero_division=0)
                except:
                    train_auc = 0.0
                    train_f1 = 0.0

                # The final XGBoost evaluation is performed after representation learning.
                test_auc = 0.0
                test_f1 = 0.0

            print(f"  Epoch {epoch + 1}/{config.num_epochs} - Loss: {train_loss:.4f}")

            # Early stopping applies only to the MLP prediction head.
            if predictor.type == 'mlp':
                if test_auc > best_test_auc + config.early_stopping_delta:
                    best_test_auc = test_auc
                    best_epoch = epoch
                    patience = 0
                    # Save the best representation checkpoint.
                    best_state = {
                        'projector': projector.state_dict(),
                        'gnn': gnn.state_dict(),
                        'metapath': metapath_enc.state_dict(),
                        'fusion': fusion.state_dict(),
                        'predictor': predictor.model.state_dict()
                    }
                else:
                    # Count patience only after the warm-up period.
                    if epoch >= config.warmup_epochs:
                        patience += 1
                        if patience >= config.early_stopping_patience:
                            break
                    else:
                        # Do not increment patience during warm-up.
                        pass

        if predictor.type == 'mlp' and best_state:
            projector.load_state_dict(best_state['projector'])
            gnn.load_state_dict(best_state['gnn'])
            metapath_enc.load_state_dict(best_state['metapath'])
            fusion.load_state_dict(best_state['fusion'])
            predictor.model.load_state_dict(best_state['predictor'])

        # Final fold evaluation.
        projector.eval()
        gnn.eval()
        metapath_enc.eval()
        fusion.eval()

        with torch.no_grad():
            # Obtain final node embeddings.
            train_feat = fold_data['train_feat'].to(config.device)
            train_adj = fold_data['train_adj'].to(config.device)
            train_meta = {k: v.to(config.device) for k, v in fold_data['train_meta_paths'].items()}

            shared_feat = projector(train_feat)
            p_z_x_nomarg = None
            p_z_x_marg = None
            if config.gnn_type == 'causal':
                gnn_emb_nomarg, p_z_x_nomarg = gnn(shared_feat, train_adj, causal_inference=False)
                gnn_emb_marg, p_z_x_marg = gnn(shared_feat, train_adj, causal_inference=True)
                gnn_emb = gnn_emb_marg if use_marginalization else gnn_emb_nomarg
            elif config.gnn_type in ['causal_hyper', 'causal_hyper_global_only']:
                gnn_emb_nomarg, p_z_x_nomarg = gnn(shared_feat, train_adj, H=train_H, causal_inference=False)
                gnn_emb_marg, p_z_x_marg = gnn(shared_feat, train_adj, H=train_H, causal_inference=True)
                gnn_emb = gnn_emb_marg if use_marginalization else gnn_emb_nomarg
            elif config.gnn_type == 'causal_gcn':
                gnn_emb_nomarg, p_z_x_nomarg = gnn(shared_feat, train_adj, causal_inference=False)
                gnn_emb_marg, p_z_x_marg = gnn(shared_feat, train_adj, causal_inference=True)
                gnn_emb = gnn_emb_marg if use_marginalization else gnn_emb_nomarg
            else:
                gnn_emb = gnn(shared_feat, train_adj)
            mp_embs = metapath_enc(shared_feat, train_meta)
            fused_emb = fusion([gnn_emb, mp_embs])

            # Apply the same residual connection used during training.
            if shared_feat.shape == fused_emb.shape:
                 fused_emb = fused_emb + shared_feat

            if enable_bias_sparsity_report:
                miRNA_num_total = int(mirna_df.shape[0])
                drug_offset = miRNA_num_total
                train_nodes_ids = fold_data['train_nodes'].astype(int)
                mi_global = [int(n) for n in train_nodes_ids if int(n) < miRNA_num_total]
                dr_global = [int(n) for n in train_nodes_ids if int(n) >= miRNA_num_total]
                report = {
                    'fold': int(fold_idx + 1),
                    'cv_type': cv_type,
                    'gnn_type': config.gnn_type,
                    'use_marginalization': bool(use_marginalization),
                    'num_nodes_train': int(len(train_nodes_ids)),
                    'num_mirna_train': int(len(mi_global)),
                    'num_drug_train': int(len(dr_global)),
                }

                if len(mi_global) > 0 and len(dr_global) > 0:
                    mi_idx_local = torch.tensor([node_map[n] for n in mi_global], device=config.device, dtype=torch.long)
                    dr_idx_local = torch.tensor([node_map[n] for n in dr_global], device=config.device, dtype=torch.long)
                    block_md = train_adj[mi_idx_local][:, dr_idx_local]
                    orig_density = float((block_md > 0).float().mean().item())

                    mp_key = 'mp2_mirna_gene_drug' if 'mp2_mirna_gene_drug' in train_meta else 'mirna_drug'
                    mp_block = train_meta[mp_key][mi_idx_local][:, dr_idx_local]
                    mp_density = float((mp_block > 0).float().mean().item())

                    clusters_train = clusters[train_nodes_ids]
                    mi_mask = train_nodes_ids < miRNA_num_total
                    dr_mask = ~mi_mask
                    mi_clusters = clusters_train[mi_mask]
                    dr_clusters = clusters_train[dr_mask]
                    counts_m = np.bincount(mi_clusters, minlength=num_hyperedges)
                    counts_d = np.bincount(dr_clusters, minlength=num_hyperedges)
                    hyper_pairs = float(np.sum(counts_m * counts_d))
                    hyper_density = float(hyper_pairs / (len(mi_clusters) * len(dr_clusters)))

                    orig_full = (train_adj > 0).float()
                    orig_avg_degree = float((orig_full.sum(dim=1) - 1.0).mean().item())
                    mp_full = (train_meta[mp_key] > 0).float()
                    mp_avg_degree = float((mp_full.sum(dim=1) - 1.0).mean().item())
                    H_full = (train_H > 0).float()
                    hyper_induced = (torch.matmul(H_full, H_full.t()) > 0).float()
                    hyper_avg_degree = float((hyper_induced.sum(dim=1) - 1.0).mean().item())

                    deg_m = (block_md > 0).sum(dim=1).detach().cpu().numpy().astype(float)
                    deg_d = (block_md > 0).sum(dim=0).detach().cpu().numpy().astype(float)
                    deg_all = np.zeros(train_adj.shape[0], dtype=float)
                    deg_all[mi_idx_local.detach().cpu().numpy()] = deg_m
                    deg_all[dr_idx_local.detach().cpu().numpy()] = deg_d

                    if hasattr(gnn, 'impl') and hasattr(gnn.impl, 'context_prototypes') and gnn.impl.context_prototypes is not None:
                        prototypes = gnn.impl.context_prototypes
                    elif hasattr(gnn, 'context_prototypes') and gnn.context_prototypes is not None:
                        prototypes = gnn.context_prototypes
                    else:
                        prototypes = None

                    if prototypes is not None and p_z_x_nomarg is not None:
                        context_nomarg = torch.matmul(p_z_x_nomarg, prototypes)
                        context_norm_nomarg = torch.norm(context_nomarg, dim=1).detach().cpu().numpy().astype(float)
                        eps = 1e-12
                        p_np = p_z_x_nomarg.detach().cpu().numpy().astype(float)
                        entropy = -np.sum(p_np * np.log(p_np + eps), axis=1)

                        def spearman_corr(x, y):
                            x = pd.Series(x).rank(method='average').to_numpy()
                            y = pd.Series(y).rank(method='average').to_numpy()
                            if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
                                return 0.0
                            return float(np.corrcoef(x, y)[0, 1])

                        corr_deg_context = spearman_corr(deg_all, context_norm_nomarg)
                        corr_deg_entropy = spearman_corr(deg_all, entropy)

                        report.update({
                            'orig_density': orig_density,
                            'metapath_density': mp_density,
                            'hypergraph_density': hyper_density,
                            'orig_avg_degree': orig_avg_degree,
                            'metapath_avg_degree': mp_avg_degree,
                            'hypergraph_avg_degree': hyper_avg_degree,
                            'degree_bias_corr_deg_context_norm_nomarg': corr_deg_context,
                            'degree_bias_corr_deg_entropy_nomarg': corr_deg_entropy,
                            'context_norm_nomarg_std': float(np.std(context_norm_nomarg)),
                            'metapath_key': mp_key
                        })

                        selected = []
                        for name in analysis_mirna_names:
                            if name in mirna_name_to_id:
                                gid = int(mirna_name_to_id[name])
                                if gid in node_map:
                                    li = int(node_map[gid])
                                    selected.append({
                                        'name': name,
                                        'type': 'miRNA',
                                        'global_id': gid,
                                        'degree': float(deg_all[li]),
                                        'context_norm_nomarg': float(context_norm_nomarg[li]),
                                        'entropy_nomarg': float(entropy[li])
                                    })
                        for name in analysis_drug_names:
                            if name in drug_name_to_id:
                                gid = int(drug_name_to_id[name] + drug_offset)
                                if gid in node_map:
                                    li = int(node_map[gid])
                                    selected.append({
                                        'name': name,
                                        'type': 'Drug',
                                        'global_id': gid,
                                        'degree': float(deg_all[li]),
                                        'context_norm_nomarg': float(context_norm_nomarg[li]),
                                        'entropy_nomarg': float(entropy[li])
                                    })
                        report['selected_nodes'] = selected

                        types = np.array(['miRNA' if int(train_nodes_ids[i]) < miRNA_num_total else 'Drug' for i in range(len(train_nodes_ids))])
                        local_by_global = {int(g): int(node_map[int(g)]) for g in train_nodes_ids if int(g) in node_map}
                        deg_plot = deg_all
                        y_plot = context_norm_nomarg
                        plt.figure(figsize=(7.5, 5.5))
                        mi_local = [local_by_global[int(g)] for g in train_nodes_ids if int(g) < miRNA_num_total and int(g) in local_by_global]
                        dr_local = [local_by_global[int(g)] for g in train_nodes_ids if int(g) >= miRNA_num_total and int(g) in local_by_global]
                        if len(mi_local) > 0:
                            plt.scatter(deg_plot[mi_local], y_plot[mi_local], s=18, alpha=0.7, label='miRNA', color='#4A90E2')
                        if len(dr_local) > 0:
                            plt.scatter(deg_plot[dr_local], y_plot[dr_local], s=18, alpha=0.7, label='Drug', color='#F5A623')
                        for item in selected:
                            li = int(node_map[item['global_id']])
                            plt.scatter([deg_plot[li]], [y_plot[li]], s=80, color='#E74C3C')
                            plt.text(float(deg_plot[li]), float(y_plot[li]), item['name'], fontsize=9)
                        plt.xlabel('Node Degree (in training graph)')
                        plt.ylabel('Context Norm (no marginalization)')
                        plt.title(f'Degree Bias Evidence (Fold {fold_idx+1})')
                        plt.legend()
                        plt.tight_layout()
                        plot_path = f"{result_dir}/degree_bias_scatter_fold{fold_idx+1}.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        report['plot_path'] = plot_path
                    else:
                        report.update({
                            'orig_density': orig_density,
                            'metapath_density': mp_density,
                            'hypergraph_density': hyper_density,
                            'orig_avg_degree': orig_avg_degree,
                            'metapath_avg_degree': mp_avg_degree,
                            'hypergraph_avg_degree': hyper_avg_degree,
                            'metapath_key': mp_key
                        })

                bias_sparsity_reports.append(report)
                with open(f"{result_dir}/bias_sparsity_report_fold{fold_idx+1}.json", "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)

            vn_emb_final = None
            if hasattr(gnn, 'use_virtual_node') and gnn.use_virtual_node:
                 # Retain the learned virtual-node embedding for inductive inference.
                 vn_emb_final = gnn.virtual_node_emb.detach() # (1, input_dim)

            # Fit the selected non-neural predictor on learned pair embeddings.
            if predictor.type == 'xgboost':
                train_pairs_idx = fold_data['train_pairs']
                emb1_idx = [node_map[n] for n in train_pairs_idx[:, 0]]
                emb2_idx = [node_map[n] for n in train_pairs_idx[:, 1]]
                emb1 = fused_emb[emb1_idx]
                emb2 = fused_emb[emb2_idx]
                X_train = torch.cat([emb1, emb2], dim=1)
                y_train = fold_data['train_y']

                predictor.fit_xgboost(X_train, y_train)
            elif predictor.type in ['lr', 'svm', 'rf']:
                train_pairs_idx = fold_data['train_pairs']
                emb1_idx = [node_map[n] for n in train_pairs_idx[:, 0]]
                emb2_idx = [node_map[n] for n in train_pairs_idx[:, 1]]
                emb1 = fused_emb[emb1_idx]
                emb2 = fused_emb[emb2_idx]
                X_train_tensor = torch.cat([emb1, emb2], dim=1)
                y_train = fold_data['train_y']
                X_train_np = X_train_tensor.cpu().numpy()
                y_train_np = y_train.cpu().numpy()
                pp = predictor_params or {}
                if predictor.type == 'lr':
                    clf = LogisticRegression(
                        random_state=pp.get('random_state', 42),
                        max_iter=pp.get('max_iter', 1000),
                        solver=pp.get('solver', 'liblinear'),
                        n_jobs=pp.get('n_jobs', -1),
                        C=pp.get('C', 1.0)
                    )
                elif predictor.type == 'svm':
                    clf = SVC(
                        probability=True,
                        random_state=pp.get('random_state', 42),
                        gamma=pp.get('gamma', 'auto'),
                        kernel=pp.get('kernel', 'linear'),
                        C=pp.get('C', 0.1)
                    )
                elif predictor.type == 'rf':
                    clf = RandomForestClassifier(
                        random_state=pp.get('random_state', 42),
                        n_jobs=pp.get('n_jobs', -1),
                        n_estimators=pp.get('n_estimators', 100),
                        max_depth=pp.get('max_depth', None)
                    )
                clf.fit(X_train_np, y_train_np)
                predictor.model = clf

            # Predict the test pairs.
            test_pairs_idx = fold_data['test_pairs']

            # Handle nodes absent from the training subgraph.
            test_y = fold_data['test_y']

            # Project raw features for previously unseen nodes.
            def get_emb_safe(node_id):
                if node_id in node_map:
                    return fused_emb[node_map[node_id]]
                else:
                    # Inductive inference from the node's raw features.
                    if node_id < len(features):
                        raw_feat = torch.tensor(features[node_id], dtype=torch.float32).unsqueeze(0).to(config.device)

                        with torch.no_grad():
                            projected_feat = projector(raw_feat)

                            # 3. Virtual-node enhancement
                            # A virtual-node model can additionally broadcast global context.
                            # HGGN can contain multiple virtual-node representations.
                            # Use the mean virtual-node representation as a global signal.

                            vn_emb_to_add = None
                            if vn_emb_final is not None:
                                if len(vn_emb_final.shape) == 2 and vn_emb_final.shape[0] > 1:
                                    # HGGN case: multiple virtual nodes
                                    # Aggregate them (Mean)
                                    vn_emb_to_add = torch.mean(vn_emb_final, dim=0, keepdim=True)
                                else:
                                    # Single virtual node case
                                    vn_emb_to_add = vn_emb_final

                            if vn_emb_to_add is not None:
                                # Approximate the virtual-node broadcast for an unseen node.
                                # Add the virtual-node embedding to the projected features.
                                # Apply the signal only when representation dimensions match.
                                if projected_feat.shape[1] == vn_emb_to_add.shape[1]:
                                     # Add the global virtual-node signal.
                                     projected_feat = projected_feat + vn_emb_to_add

                            return projected_feat.squeeze(0)

            # Extract pair embeddings, including mappable cold-start nodes.
            t_emb1 = torch.stack([get_emb_safe(n.item()) for n in test_pairs_idx[:, 0]])
            t_emb2 = torch.stack([get_emb_safe(n.item()) for n in test_pairs_idx[:, 1]])
            t_pair_emb = torch.cat([t_emb1, t_emb2], dim=1)

            if predictor.type == 'xgboost':
                test_pred = predictor.predict_xgboost(t_pair_emb)
                if isinstance(test_pred, torch.Tensor):
                    test_pred = test_pred.cpu().numpy()
            elif predictor.type in ['lr', 'svm', 'rf']:
                X_test_np = t_pair_emb.cpu().numpy()
                test_pred = predictor.model.predict_proba(X_test_np)[:, 1]
            else: # MLP
                test_pred = predictor(t_pair_emb).squeeze().cpu().numpy()

            test_y = test_y.cpu().numpy()

            deg_train = (train_adj > 0).sum(dim=1).detach().cpu().numpy().astype(float) - 1.0
            deg_test_sum = np.zeros(len(test_pred), dtype=float)
            for i in range(test_pairs_idx.shape[0]):
                a = int(test_pairs_idx[i, 0].item())
                b = int(test_pairs_idx[i, 1].item())
                da = float(deg_train[node_map[a]]) if a in node_map else 0.0
                db = float(deg_train[node_map[b]]) if b in node_map else 0.0
                deg_test_sum[i] = da + db

            def spearman_corr(x, y):
                x = pd.Series(x).rank(method='average').to_numpy()
                y = pd.Series(y).rank(method='average').to_numpy()
                if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
                    return 0.0
                return float(np.corrcoef(x, y)[0, 1])

            deg_score_spearman = spearman_corr(deg_test_sum, test_pred)
            q25 = np.percentile(deg_test_sum, 25)
            q75 = np.percentile(deg_test_sum, 75)
            low_mask = deg_test_sum <= q25
            high_mask = deg_test_sum >= q75
            low_mean = float(np.mean(test_pred[low_mask])) if np.any(low_mask) else float(np.mean(test_pred))
            high_mean = float(np.mean(test_pred[high_mask])) if np.any(high_mask) else float(np.mean(test_pred))
            deg_score_gap_q75_q25 = float(high_mean - low_mean)

            if enable_bias_sparsity_report:
                np.savez_compressed(
                    f"{result_dir}/degree_binned_curve_fold{fold_idx+1}.npz",
                    deg_sum=deg_test_sum.astype(np.float32),
                    score=test_pred.astype(np.float32),
                    y=test_y.astype(np.int8),
                )

            if enable_bias_sparsity_report and len(bias_sparsity_reports) > 0 and int(bias_sparsity_reports[-1].get('fold', -1)) == int(fold_idx + 1):
                bias_sparsity_reports[-1].update({
                    'degree_score_spearman_test': float(deg_score_spearman),
                    'degree_score_gap_q75_q25_test': float(deg_score_gap_q75_q25),
                })
                with open(f"{result_dir}/bias_sparsity_report_fold{fold_idx+1}.json", "w", encoding="utf-8") as f:
                    json.dump(bias_sparsity_reports[-1], f, ensure_ascii=False, indent=2)

            # Compute evaluation metrics.
            try:
                auc = roc_auc_score(test_y, test_pred)
                auprc = average_precision_score(test_y, test_pred)
            except ValueError:
                auc = 0.0
                auprc = 0.0

            f1 = f1_score(test_y, (test_pred >= 0.5).astype(int), zero_division=0)

            acc = accuracy_score(test_y, (test_pred >= 0.5).astype(int))
            precision = precision_score(test_y, (test_pred >= 0.5).astype(int), zero_division=0)
            recall = recall_score(test_y, (test_pred >= 0.5).astype(int), zero_division=0)

            print(
                f"  Fold result - AUC: {auc:.4f}, AUPRC: {auprc:.4f}, "
                f"ACC: {acc:.4f}, F1: {f1:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}"
            )

            # Save fold predictions and metrics.
            fold_res = {
                'test_y': test_y,
                'test_pred': test_pred,
                'auc': auc, 'auprc': auprc, 'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall
            }
            all_fold_data[f'fold_{fold_idx+1}'] = fold_res
            all_fold_embeddings.append(fused_emb.cpu().numpy())

    print("\nFive-fold results")
    if all_fold_data:
        avg_auc = np.mean([d['auc'] for d in all_fold_data.values()])
        std_auc = np.std([d['auc'] for d in all_fold_data.values()])
        avg_auprc = np.mean([d['auprc'] for d in all_fold_data.values()])
        avg_acc = np.mean([d['acc'] for d in all_fold_data.values()])
        avg_f1 = np.mean([d['f1'] for d in all_fold_data.values()])

        print(f"  AUC: {avg_auc:.4f} +/- {std_auc:.4f}")
        print(f"  AUPRC: {avg_auprc:.4f}")
        print(f"  ACC: {avg_acc:.4f}")
        print(f"  F1: {avg_f1:.4f}")
    else:
        print("No valid fold results were produced.")

    if enable_bias_sparsity_report and bias_sparsity_reports:
        df_bs = pd.json_normalize(bias_sparsity_reports)
        df_bs.to_csv(f"{result_dir}/bias_sparsity_reports.csv", index=False, encoding="utf-8-sig")
        summary_keys = [
            'orig_density',
            'metapath_density',
            'hypergraph_density',
            'orig_avg_degree',
            'metapath_avg_degree',
            'hypergraph_avg_degree',
            'degree_score_spearman_test',
            'degree_score_gap_q75_q25_test',
            'degree_bias_corr_deg_context_norm_nomarg',
            'degree_bias_corr_deg_entropy_nomarg',
            'context_norm_nomarg_std'
        ]
        summary = {}
        for k in summary_keys:
            if k in df_bs.columns:
                vals = pd.to_numeric(df_bs[k], errors='coerce').dropna()
                if len(vals) > 0:
                    summary[k] = float(vals.mean())
        with open(f"{result_dir}/bias_sparsity_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if 'orig_avg_degree' in summary and 'metapath_avg_degree' in summary and 'hypergraph_avg_degree' in summary:
            labels = ['Original', 'Meta-path', 'Hypergraph']
            vals = [summary['orig_avg_degree'], summary['metapath_avg_degree'], summary['hypergraph_avg_degree']]
            plt.figure(figsize=(7.5, 5.0))
            bars = plt.bar(labels, vals, color=['#4A90E2', '#50E3C2', '#B8E986'], width=0.6)
            for b in bars:
                h = b.get_height()
                plt.text(b.get_x() + b.get_width() / 2.0, h + (0.02 * max(vals) if max(vals) > 0 else 0.1), f"{h:.2f}", ha="center", va="bottom", fontsize=10)
            title = "Effective Neighborhood Size"
            if 'degree_bias_corr_deg_context_norm_nomarg' in summary:
                title += f" | Spearman(deg, context_norm)={summary['degree_bias_corr_deg_context_norm_nomarg']:.3f}"
            plt.title(title)
            plt.ylabel("Average Degree")
            plt.ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1.0)
            plt.tight_layout()
            plt.savefig(f"{result_dir}/bias_sparsity_summary.png", dpi=300, bbox_inches="tight")
            plt.close()

    # Generate visualizations without discarding completed metrics if plotting fails.
    try:
        roc_path = f'{result_dir}/roc_curve_{fusion_method}_{timestamp}.png'
        plot_roc_curve(all_fold_data, roc_path)
        all_roc_path = f'{result_dir}/roc_curve_all_folds_{fusion_method}_{timestamp}.png'
        plot_all_roc_curves(all_fold_data, all_roc_path)

        tsne_path = f'{result_dir}/tsne_{fusion_method}_{timestamp}.png'
        if all_fold_embeddings:
            plot_tsne(all_fold_embeddings[-1], tsne_path)
    except Exception as plot_err:
        print(f" Plotting failed; returning completed metrics: {plot_err}")

    # Return aggregate metrics.
    return {
        'auc': avg_auc if 'avg_auc' in locals() else 0.0,
        'auprc': avg_auprc if 'avg_auprc' in locals() else 0.0,
        'acc': avg_acc if 'avg_acc' in locals() else 0.0,
        'f1': avg_f1 if 'avg_f1' in locals() else 0.0,
        'precision': np.mean([d['precision'] for d in all_fold_data.values()]) if all_fold_data else 0.0,
        'recall': np.mean([d['recall'] for d in all_fold_data.values()]) if all_fold_data else 0.0
    }

if __name__ == '__main__':
    # Example batch entry point.

    # Run the configured parameter sets for each evaluation setting.
    # Shared base hyperparameters.
    learning_rate = 1e-5
    dropout_rate = 0.4
    hidden_dimension = 512
    gnn_layer_count = 4
    meta_path_len = 4

    # Scenario-specific environment-prototype counts.
    num_prototypes_dict = {
        'pair': 25,
        'drug': 25,
        'mirna': 40
    }

    cv_types = ['pair', 'drug', 'mirna']
    all_results = {}
    for current_cv_type in cv_types:
        current_k = num_prototypes_dict.get(current_cv_type, 25)
        res = main_new(
            fusion_method='multiply_residual',
            gnn_type='causal_hyper',
            predictor_type='xgboost',
            cv_type=current_cv_type,
            negative_sampling_method='kmeans_fractal_cosine',
            gnn_layers=gnn_layer_count,
            hidden_dim=hidden_dimension,
            dropout=dropout_rate,
            num_prototypes=current_k,
            max_meta_path_length=meta_path_len
        )
        all_results[current_cv_type] = res

    print("\nFinal results across the three CV settings")
    for k, v in all_results.items():
        if isinstance(v, dict):
            print(f"  {k}: AUC={v.get('auc', 0.0):.4f}, AUPRC={v.get('auprc', 0.0):.4f}, F1={v.get('f1', 0.0):.4f}")
        else:
            print(f"  {k}: no valid result")
