"""
========================================
 Feature loading and fusion
========================================

Load, fuse, and save structural and semantic node features.
"""

import numpy as np
import pandas as pd


def load_features_and_concatenate(data_dir, fusion_method='concatenate'):
    """Load structural and semantic features and apply the selected fusion method."""
    # Structural feature files.
    miRNA_struct_csv = f"{data_dir}/1_miRNA_sequence.csv"
    drug_struct_csv = f"{data_dir}/1_drug_smiles.csv"

    # BioBERT semantic feature files.
    miRNA_semantic_npz = f"{data_dir}/miRNA_BioBERT_semantic_features_128d.npz"
    drug_semantic_npz = f"{data_dir}/drug_BioBERT_semantic_features_128d.npz"

    # Load 128-dimensional miRNA structural features.
    miRNA_struct_df = pd.read_csv(miRNA_struct_csv, encoding="utf-8-sig")
    miRNA_struct_feat = miRNA_struct_df[[f'mirna_feat_{i}' for i in range(128)]].values

    # Load 128-dimensional drug structural features.
    drug_struct_df = pd.read_csv(drug_struct_csv, encoding="utf-8-sig")
    drug_struct_feat = drug_struct_df[[f'struct_feat_{i}' for i in range(128)]].values

    # Load 128-dimensional BioBERT semantic features.
    miRNA_semantic_feat = np.load(miRNA_semantic_npz)['feature']
    drug_semantic_feat = np.load(drug_semantic_npz)['feature']

    def fuse_features(struct_feat, semantic_feat, method):
        """Fuse two feature matrices."""
        if method == 'concatenate':
            # Direct concatenation (256 dimensions).
            return np.concatenate([struct_feat, semantic_feat], axis=1)
        elif method == 'add':
            # Element-wise addition (128 dimensions).
            return struct_feat + semantic_feat
        elif method == 'average':
            # Element-wise average (128 dimensions).
            return (struct_feat + semantic_feat) / 2
        elif method == 'multiply':
            # Element-wise multiplication (128 dimensions).
            return struct_feat * semantic_feat
        elif method == 'attention':
            # Lightweight attention fusion (128 dimensions).
            weights = np.exp(struct_feat.sum(axis=1, keepdims=True)) / \
                      (np.exp(struct_feat.sum(axis=1, keepdims=True)) + np.exp(semantic_feat.sum(axis=1, keepdims=True)))
            return weights * struct_feat + (1 - weights) * semantic_feat
        elif method == 'mlp':
            # Fixed linear-projection baseline (128 dimensions).
            combined = np.concatenate([struct_feat, semantic_feat], axis=1)
            # The full model uses the trainable FusionModule instead.
            np.random.seed(42)
            weights = np.random.randn(256, 128) * 0.01
            bias = np.zeros(128)
            return combined.dot(weights) + bias
        elif method == 'pca':
            # PCA fusion (128 dimensions).
            from sklearn.decomposition import PCA
            combined = np.concatenate([struct_feat, semantic_feat], axis=1)
            pca = PCA(n_components=128, random_state=42)
            return pca.fit_transform(combined)
        elif method == 'neural':
            # Fixed two-layer neural baseline (128 dimensions).
            combined = np.concatenate([struct_feat, semantic_feat], axis=1)
            # First layer.
            np.random.seed(42)
            w1 = np.random.randn(256, 128) * 0.01
            b1 = np.zeros(128)
            hidden = np.maximum(0, combined.dot(w1) + b1)
            # Second layer.
            w2 = np.random.randn(128, 128) * 0.01
            b2 = np.zeros(128)
            return hidden.dot(w2) + b2
        else:
            raise ValueError(f"Unsupported fusion method: {method}")

    # Apply the same operation to both entity types.
    miRNA_final_feature = fuse_features(miRNA_struct_feat, miRNA_semantic_feat, fusion_method)
    drug_final_feature = fuse_features(drug_struct_feat, drug_semantic_feat, fusion_method)

    # Save fused features.
    feature_dim = miRNA_final_feature.shape[1]
    np.savez(f"{data_dir}/miRNA_fused_node_features_{feature_dim}d.npz", feature=miRNA_final_feature)
    np.savez(f"{data_dir}/drug_fused_node_features_{feature_dim}d.npz", feature=drug_final_feature)

    return miRNA_final_feature, drug_final_feature
