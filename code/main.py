"""
========================================
 Core configuration and negative-sampling utilities
========================================

Provides sampling strategies and shared configuration.
"""

import numpy as np
import torch
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '4'
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


def generate_negative_pairs_random(positive_pairs, num_nodes, num_neg):
    """Generate random negative pairs."""
    neg_pairs = []
    while len(neg_pairs) < num_neg:
        n1 = np.random.randint(0, num_nodes)
        n2 = np.random.randint(0, num_nodes)
        if n1 != n2 and not any((positive_pairs[:, 0] == n1) & (positive_pairs[:, 1] == n2)):
            neg_pairs.append((n1, n2))
    neg_pairs_array = np.array(neg_pairs)
    return neg_pairs_array


def generate_negative_pairs_kmeans_original_cosine(positive_pairs, features, num_neg):
    """Generate negatives with the baseline KMeans-cosine strategy."""
    num_nodes = features.shape[0]
    neg_pairs = []

    n_clusters = 10

    features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(features_normalized)

    cluster_nodes = {i: np.where(labels == i)[0] for i in range(n_clusters)}

    max_attempts = num_neg * 10
    attempts = 0

    while len(neg_pairs) < num_neg and attempts < max_attempts:
        attempts += 1

        if np.random.random() < 0.7:
            c1, c2 = np.random.choice(n_clusters, 2, replace=False)
        else:
            c1 = c2 = np.random.choice(n_clusters)

        if len(cluster_nodes[c1]) > 0 and len(cluster_nodes[c2]) > 0:
            n1 = np.random.choice(cluster_nodes[c1])
            n2 = np.random.choice(cluster_nodes[c2])

            if n1 != n2 and not any((positive_pairs[:, 0] == n1) & (positive_pairs[:, 1] == n2)):
                neg_pairs.append((n1, n2))

    if len(neg_pairs) < num_neg:
        while len(neg_pairs) < num_neg:
            n1 = np.random.randint(0, num_nodes)
            n2 = np.random.randint(0, num_nodes)
            if n1 != n2 and not any((positive_pairs[:, 0] == n1) & (positive_pairs[:, 1] == n2)):
                neg_pairs.append((n1, n2))

    neg_pairs_array = np.array(neg_pairs)
    return neg_pairs_array


def calculate_cluster_cohesion(features, labels):
    """Compute mean within-cluster cosine similarity."""
    n_clusters = len(np.unique(labels))
    cohesion = {}

    features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    for c in range(n_clusters):
        cluster_samples = features_normalized[labels == c]
        if len(cluster_samples) > 1:
            sim_matrix = cosine_similarity(cluster_samples)
            upper_tri = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
            cohesion[c] = np.mean(upper_tri) if len(upper_tri) > 0 else 0.0
        else:
            cohesion[c] = 0.0

    return cohesion


def calculate_local_fractal_dimension(features, k=10):
    """Estimate local intrinsic dimension in feature space."""
    from sklearn.neighbors import NearestNeighbors
    features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    k = min(k, features.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(features_normalized)
    distances, _ = nn.kneighbors(features_normalized)

    r_k = distances[:, -1] + 1e-8

    lid = np.zeros(features.shape[0])
    for i in range(features.shape[0]):
        r_i = distances[i, 1:k+1] + 1e-8
        ratio = r_k[i] / r_i
        ratio = np.clip(ratio, 1.0, None)
        log_ratio = np.log(ratio)
        mean_log = np.mean(log_ratio)
        if mean_log > 0:
            lid[i] = 1.0 / mean_log
        else:
            lid[i] = 0.0

    lid_norm = (lid - np.min(lid)) / (np.max(lid) - np.min(lid) + 1e-8)
    weights = lid_norm + 0.1
    return weights


def generate_negative_pairs_kmeans(positive_pairs, features, num_neg, distance_metric='euclidean',
                                   optimized_init=False, dynamic_sampling=False,
                                   hybrid_distance=False, parallel=False, adj_matrix=None,
                                   kmeans_model=None, use_fractal_dimension=False):
    """Generate negatives with KMeans clustering and optional LID weighting."""
    num_nodes = features.shape[0]
    neg_pairs = []

    n_clusters = 10

    node_weights = None
    if use_fractal_dimension:
        node_weights = calculate_local_fractal_dimension(features)

    if kmeans_model is not None:
        if distance_metric == 'cosine':
            features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            labels = kmeans_model.predict(features_normalized)
        elif hybrid_distance:
            feature_dim = features.shape[1]
            split_point = feature_dim // 2
            features_euclidean = features[:, :split_point]
            features_cosine = features[:, split_point:]
            features_cosine_normalized = features_cosine / (np.linalg.norm(features_cosine, axis=1, keepdims=True) + 1e-8)
            hybrid_features = np.hstack([features_euclidean, features_cosine_normalized])
            labels = kmeans_model.predict(hybrid_features)
        else:
            feature_variances = np.var(features, axis=0)
            high_var_features = features[:, feature_variances > np.percentile(feature_variances, 20)]
            if high_var_features.shape[1] == 0:
                high_var_features = features
            labels = kmeans_model.predict(high_var_features)
    else:
        if hybrid_distance:
            feature_dim = features.shape[1]
            split_point = feature_dim // 2

            features_euclidean = features[:, :split_point]
            features_cosine = features[:, split_point:]
            features_cosine_normalized = features_cosine / (np.linalg.norm(features_cosine, axis=1, keepdims=True) + 1e-8)

            hybrid_features = np.hstack([features_euclidean, features_cosine_normalized])

            init_method = 'k-means++' if optimized_init else 'random'
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=init_method,
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(hybrid_features)

        elif distance_metric == 'cosine':
            features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

            kmeans = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(features_normalized)

        else:

            feature_variances = np.var(features, axis=0)
            high_var_features = features[:, feature_variances > np.percentile(feature_variances, 20)]

            if high_var_features.shape[1] == 0:
                high_var_features = features

            init_method = 'k-means++' if optimized_init else 'random'
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=init_method,
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(high_var_features)

    cluster_nodes = {i: np.where(labels == i)[0] for i in range(n_clusters)}

    max_attempts = num_neg * 10
    attempts = 0

    while len(neg_pairs) < num_neg and attempts < max_attempts:
        attempts += 1

        if np.random.random() < 0.6:
            c1, c2 = np.random.choice(n_clusters, 2, replace=False)
        else:
            c1 = c2 = np.random.choice(n_clusters)

        if len(cluster_nodes[c1]) > 0 and len(cluster_nodes[c2]) > 0:
            if use_fractal_dimension and c1 == c2:
                nodes_c1 = cluster_nodes[c1]
                w_c1 = node_weights[nodes_c1]
                p_c1 = w_c1 / np.sum(w_c1)
                n1 = np.random.choice(nodes_c1, p=p_c1)
                n2 = np.random.choice(nodes_c1, p=p_c1)
            else:
                n1 = np.random.choice(cluster_nodes[c1])
                n2 = np.random.choice(cluster_nodes[c2])

            if n1 != n2 and not any((positive_pairs[:, 0] == n1) & (positive_pairs[:, 1] == n2)):
                neg_pairs.append((n1, n2))

    if len(neg_pairs) < num_neg:
        max_random_supplement = int(num_neg * 0.2)
        supplement_count = 0

        while len(neg_pairs) < num_neg and supplement_count < max_random_supplement:
            supplement_count += 1
            n1 = np.random.randint(0, num_nodes)
            n2 = np.random.randint(0, num_nodes)
            if n1 != n2 and not any((positive_pairs[:, 0] == n1) & (positive_pairs[:, 1] == n2)):
                neg_pairs.append((n1, n2))

    if len(neg_pairs) < num_neg:
        k_buffer = 10
        while len(neg_pairs) < num_neg:
            pos_pair = positive_pairs[np.random.randint(len(positive_pairs))]
            n1 = pos_pair[0]
            distances = cosine_distances(features[n1:n1+1], features)[0]
            sorted_indices = np.argsort(distances)

            candidates = sorted_indices[1:k_buffer+1]
            np.random.shuffle(candidates)

            for n2 in candidates:
                if n1 != n2 and not any((positive_pairs[:, 0] == n1) & (positive_pairs[:, 1] == n2)):
                    neg_pairs.append((n1, n2))
                    break

    neg_pairs_array = np.array(neg_pairs)

    if kmeans_model is None:
        return neg_pairs_array, kmeans
    else:
        return neg_pairs_array

def generate_negative_pairs_structural(positive_pairs, adj_matrix, num_neg):
    """Generate negatives from graph structure."""
    num_nodes = adj_matrix.shape[0]
    neg_pairs = []

    degrees = np.sum(adj_matrix, axis=1)

    degree_probs = degrees / np.sum(degrees)

    while len(neg_pairs) < num_neg:
        n1 = np.random.choice(num_nodes, p=degree_probs)
        n2 = np.random.choice(num_nodes, p=degree_probs)

        if n1 != n2 and not any((positive_pairs[:, 0] == n1) & (positive_pairs[:, 1] == n2)):
            if adj_matrix[n1, n2] == 0:
                neg_pairs.append((n1, n2))

    return np.array(neg_pairs)

def generate_negative_pairs_hard(positive_pairs, features, num_neg):
    """Generate feature-similar hard negatives."""
    num_nodes = features.shape[0]
    neg_pairs = []

    while len(neg_pairs) < num_neg:
        pos_pair = positive_pairs[np.random.randint(len(positive_pairs))]
        n1 = pos_pair[0]

        distances = np.linalg.norm(features - features[n1], axis=1)
        sorted_indices = np.argsort(distances)

        for n2 in sorted_indices:
            if n1 != n2 and not any((positive_pairs[:, 0] == n1) & (positive_pairs[:, 1] == n2)):
                neg_pairs.append((n1, n2))
                if len(neg_pairs) >= num_neg:
                    break

    return np.array(neg_pairs)

def generate_negative_pairs(positive_pairs, num_nodes, num_neg, method='random', features=None, adj_matrix=None, distance_metric='euclidean', kmeans_model=None):
    """Dispatch to the selected negative-sampling strategy."""
    if method == 'random':
        return generate_negative_pairs_random(positive_pairs, num_nodes, num_neg)
    elif method == 'kmeans_euclidean':
        result = generate_negative_pairs_kmeans(positive_pairs, features, num_neg, distance_metric='euclidean', adj_matrix=adj_matrix, kmeans_model=kmeans_model)
        return result
    elif method == 'kmeans_cosine':
        return generate_negative_pairs_kmeans_original_cosine(positive_pairs, features, num_neg)
    elif method == 'kmeans_optimized_euclidean':
        result = generate_negative_pairs_kmeans(
            positive_pairs, features, num_neg,
            distance_metric='euclidean',
            optimized_init=True,
            dynamic_sampling=True,
            parallel=True,
            adj_matrix=adj_matrix,
            kmeans_model=kmeans_model
        )
        return result
    elif method == 'kmeans_optimized_cosine':
        result = generate_negative_pairs_kmeans(
            positive_pairs, features, num_neg,
            distance_metric='cosine',
            optimized_init=True,
            dynamic_sampling=True,
            parallel=True,
            adj_matrix=adj_matrix,
            kmeans_model=kmeans_model
        )
        return result
    elif method == 'kmeans_optimized_hybrid':
        result = generate_negative_pairs_kmeans(
            positive_pairs, features, num_neg,
            hybrid_distance=True,
            optimized_init=True,
            dynamic_sampling=True,
            parallel=True,
            adj_matrix=adj_matrix,
            kmeans_model=kmeans_model
        )
        return result
    elif method == 'kmeans_fractal_cosine':
        result = generate_negative_pairs_kmeans(
            positive_pairs, features, num_neg,
            distance_metric='cosine',
            optimized_init=True,
            dynamic_sampling=True,
            parallel=True,
            adj_matrix=adj_matrix,
            kmeans_model=kmeans_model,
            use_fractal_dimension=True
        )
        return result
    elif method == 'structural':
        return generate_negative_pairs_structural(positive_pairs, adj_matrix, num_neg)
    elif method == 'hard':
        return generate_negative_pairs_hard(positive_pairs, features, num_neg)
    else:
        raise ValueError(f"Unknown negative-sampling method: {method}")


class Config:
    """Shared model configuration."""
    def __init__(self):
        self.output_dim = 32
        self.dropout = 0.6
        self.num_epochs = 150
        self.learning_rate = 0.0005
        self.weight_decay = 0.001
        self.seed = 42

        self.early_stopping_patience = 20
        self.early_stopping_delta = 0.0001

        self.batch_size = 256
        self.num_folds = 5

        self.negative_sampling_method = 'random'
        self.kmeans_n_clusters = 10

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
