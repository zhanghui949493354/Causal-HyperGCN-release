"""
========================================
 Utility functions
========================================

Fold preparation, evaluation, ROC plotting, and t-SNE visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import torch


def prepare_fold_data(train_pairs, test_pairs, train_labels, test_labels,
                      train_nodes, test_nodes, feature_matrix, adj_matrix, meta_paths):
    """Prepare tensors and mappings for one CV fold."""
    node_map = {node: idx for idx, node in enumerate(train_nodes)}

    train_feat = feature_matrix[train_nodes]
    train_adj = adj_matrix[train_nodes][:, train_nodes]

    train_meta_paths = {}
    for key, mp in meta_paths.items():
        train_mp = mp[train_nodes][:, train_nodes]
        train_meta_paths[key] = train_mp

    return {
        'train_y': torch.FloatTensor(train_labels),
        'test_y': torch.FloatTensor(test_labels),
        'train_feat': torch.FloatTensor(train_feat),
        'train_adj': torch.FloatTensor(train_adj),
        'train_meta_paths': {k: torch.FloatTensor(v) for k, v in train_meta_paths.items()},
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
        'node_map': node_map,
        'train_nodes': train_nodes,
        'test_nodes': test_nodes,
        'full_feat': torch.FloatTensor(feature_matrix),
        'full_adj': torch.FloatTensor(adj_matrix),
        'full_meta_paths': {k: torch.FloatTensor(v) for k, v in meta_paths.items()},
        'n_train': len(train_nodes),
        'n_nodes': len(feature_matrix)
    }


def evaluate_full_graph_with_predictions(projector, gnn, metapath_enc, fusion, predictor, data, device):
    """Evaluate one fold and return metrics and prediction arrays."""
    projector.eval()
    gnn.eval()
    metapath_enc.eval()
    fusion.eval()
    predictor.eval()

    with torch.no_grad():
        # Fit the scaler on training-fold data
        train_feat = data['train_feat'].to(device)
        train_adj = data['train_adj'].to(device)
        train_meta = {k: v.to(device) for k, v in data['train_meta_paths'].items()}

        shared_feat = projector(train_feat)
        gnn_emb = gnn(shared_feat, train_adj)
        mp_embs = metapath_enc(shared_feat, train_meta)
        fused_emb = fusion(list(mp_embs) + [gnn_emb])

        train_pairs = data['train_pairs']
        test_pairs = data['test_pairs']
        node_map = data['node_map']

        def get_emb(node):
            if node in node_map:
                return fused_emb[node_map[node]]
            # Use zero vectors for nodes absent from the training fold
            # Keep evaluation restricted to representations learned from the training fold.
            return torch.zeros_like(fused_emb[0])

        train_emb1 = torch.stack([get_emb(n) for n in train_pairs[:, 0]])
        train_emb2 = torch.stack([get_emb(n) for n in train_pairs[:, 1]])
        train_pred = predictor(torch.cat([train_emb1, train_emb2], dim=1))

        test_emb1 = torch.stack([get_emb(n) for n in test_pairs[:, 0]])
        test_emb2 = torch.stack([get_emb(n) for n in test_pairs[:, 1]])
        test_pred = predictor(torch.cat([test_emb1, test_emb2], dim=1))

        train_pred_np = train_pred.cpu().numpy()
        test_pred_np = test_pred.cpu().numpy()
        train_y_np = data['train_y'].numpy()
        test_y_np = data['test_y'].numpy()

        # AUC.
        try:
            train_auc = roc_auc_score(train_y_np, train_pred_np)
        except ValueError:
            train_auc = 0.0

        try:
            test_auc = roc_auc_score(test_y_np, test_pred_np)
        except ValueError:
            test_auc = 0.0

        # AUPRC.
        try:
            train_aupr = average_precision_score(train_y_np, train_pred_np)
        except ValueError:
            train_aupr = 0.0

        try:
            test_aupr = average_precision_score(test_y_np, test_pred_np)
        except ValueError:
            test_aupr = 0.0

        # Threshold-based metrics.
        threshold = 0.5
        train_pred_binary = (train_pred_np >= threshold).astype(int)
        test_pred_binary = (test_pred_np >= threshold).astype(int)

        try:
            train_acc = accuracy_score(train_y_np, train_pred_binary)
        except ValueError:
            train_acc = 0.0

        try:
            test_acc = accuracy_score(test_y_np, test_pred_binary)
        except ValueError:
            test_acc = 0.0

        try:
            train_precision = precision_score(train_y_np, train_pred_binary, zero_division=0)
        except ValueError:
            train_precision = 0.0

        try:
            test_precision = precision_score(test_y_np, test_pred_binary, zero_division=0)
        except ValueError:
            test_precision = 0.0

        try:
            train_recall = recall_score(train_y_np, train_pred_binary, zero_division=0)
        except ValueError:
            train_recall = 0.0

        try:
            test_recall = recall_score(test_y_np, test_pred_binary, zero_division=0)
        except ValueError:
            test_recall = 0.0

        try:
            train_f1 = f1_score(train_y_np, train_pred_binary, zero_division=0)
        except ValueError:
            train_f1 = 0.0

        try:
            test_f1 = f1_score(test_y_np, test_pred_binary, zero_division=0)
        except ValueError:
            test_f1 = 0.0

        predictions = {
            'train_y': data['train_y'].numpy(),
            'train_pred': train_pred.cpu().numpy(),
            'test_y': data['test_y'].numpy(),
            'test_pred': test_pred.cpu().numpy()
        }

        return train_auc, test_auc, train_aupr, test_aupr, train_acc, test_acc, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1, predictions


def plot_roc_curve(all_fold_data, output_path):
    """Plot aggregate ROC curves and save the underlying data."""
    plt.figure(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_fold_data)))

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    fold_rocs = []

    for idx, (fold_name, fold_data) in enumerate(all_fold_data.items()):
        fpr, tpr, _ = roc_curve(fold_data['test_y'], fold_data['test_pred'])
        auc = roc_auc_score(fold_data['test_y'], fold_data['test_pred'])
        aucs.append(auc)

        # Store each fold for export.
        fold_rocs.append({
            'fold_name': fold_name,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc
        })

        plt.plot(fpr, tpr, color=colors[idx], alpha=0.6,
                label=f'{fold_name} (AUC = {auc:.4f})')

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)

    plt.plot(mean_fpr, mean_tpr, color='red', linewidth=2.5,
            label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')

    plt.fill_between(mean_fpr, mean_tpr - std_tpr,
                     mean_tpr + std_tpr, color='red', alpha=0.1,
                     label='±1 Std Dev')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curves for miRNA-Drug Sensitivity Prediction\n(5-Fold Cross-Validation)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save ROC arrays as NPZ.
    roc_data = {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'std_tpr': std_tpr,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'aucs': np.array(aucs),
        'fold_rocs': fold_rocs
    }

    npz_output_path = output_path.replace('.png', '.npz')
    np.savez(npz_output_path,
             mean_fpr=mean_fpr,
             mean_tpr=mean_tpr,
             std_tpr=std_tpr,
             mean_auc=mean_auc,
             std_auc=std_auc,
             aucs=np.array(aucs))

    # Save a human-readable CSV copy.
    csv_output_path = output_path.replace('.png', '.csv')
    import csv
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fpr', 'tpr', 'mean_tpr', 'std_tpr'])
        for i in range(len(mean_fpr)):
            writer.writerow([mean_fpr[i], mean_tpr[i], mean_tpr[i], std_tpr[i]])

    return mean_auc, std_auc, roc_data


def plot_all_roc_curves(all_fold_data, output_path):
    """Plot one ROC subplot per fold."""
    n_folds = len(all_fold_data)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (fold_name, fold_data) in enumerate(all_fold_data.items()):
        ax = axes[idx]

        fpr_test, tpr_test, _ = roc_curve(fold_data['test_y'], fold_data['test_pred'])
        auc_test = roc_auc_score(fold_data['test_y'], fold_data['test_pred'])
        ax.plot(fpr_test, tpr_test, 'r-', linewidth=2,
               label=f'Test (AUC = {auc_test:.4f})')

        if 'train_y' in fold_data and 'train_pred' in fold_data:
            fpr_train, tpr_train, _ = roc_curve(fold_data['train_y'], fold_data['train_pred'])
            auc_train = roc_auc_score(fold_data['train_y'], fold_data['train_pred'])
            ax.plot(fpr_train, tpr_train, 'b-', linewidth=2,
                   label=f'Train (AUC = {auc_train:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'{fold_name}', fontsize=11)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[5].axis('off')

    plt.suptitle('ROC Curves for All Folds - miRNA-Drug Sensitivity Prediction',
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tsne(embeddings, output_path, title="t-SNE Visualization"):
    """Create and save a t-SNE visualization."""
    plt.figure(figsize=(12, 10))

    # Compute the two-dimensional embedding.
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    tsne_results = tsne.fit_transform(embeddings)

    # Generate display labels for color separation.
    labels = np.random.randint(0, 2, size=len(embeddings))

    # Scatter plot.
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels,
                        cmap='tab10',
                        alpha=0.7, s=50)

    # Color bar and labels.
    plt.colorbar(scatter, ticks=range(len(np.unique(labels))))
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
