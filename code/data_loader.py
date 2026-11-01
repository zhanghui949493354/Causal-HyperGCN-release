"""
========================================
 Data loading and meta-path construction
========================================

Load biological inputs, align entities, and construct meta-path matrices.
"""

import numpy as np
import pandas as pd
import os
from feature_fusion import load_features_and_concatenate


def remove_drug_associations(positive_pairs, drug_name, drug_name_to_id, miRNA_num):
    """Remove all positive associations involving one drug."""
    if drug_name not in drug_name_to_id:
        print(f" Drug not found in the dataset: {drug_name}")
        return positive_pairs, np.array([]), 0

    drug_offset = miRNA_num
    drug_id = drug_name_to_id[drug_name]
    target_drug_node_id = drug_id + drug_offset

    # Select associations involving the target drug.
    mask = positive_pairs[:, 1] != target_drug_node_id
    modified_pairs = positive_pairs[mask]
    removed_pairs = positive_pairs[~mask]
    removed_count = len(removed_pairs)

    return modified_pairs, removed_pairs, removed_count


def remove_mirna_associations(positive_pairs, mirna_name, mirna_name_to_id):
    """Remove all positive associations involving one miRNA."""
    if mirna_name not in mirna_name_to_id:
        print(f" miRNA not found in the dataset: {mirna_name}")
        return positive_pairs, np.array([]), 0

    mirna_id = mirna_name_to_id[mirna_name]

    # Select associations involving the target miRNA.
    mask = positive_pairs[:, 0] != mirna_id
    modified_pairs = positive_pairs[mask]
    removed_pairs = positive_pairs[~mask]
    removed_count = len(removed_pairs)

    return modified_pairs, removed_pairs, removed_count


def load_biological_data_with_metapaths(data_dir, fusion_method='concatenate', max_meta_path_length=4):
    """Load biological data and construct meta-path matrices."""
    drug_gene_path = f"{data_dir}/drug_gene_matrix.csv"
    mirna_gene_path = f"{data_dir}/miRNA_gene_matrix.csv"

    # Load and fuse node features.
    miRNA_feat, drug_feat = load_features_and_concatenate(data_dir, fusion_method)

    miRNA_num = miRNA_feat.shape[0]
    drug_num = drug_feat.shape[0]
    n_nodes = miRNA_num + drug_num
    drug_offset = miRNA_num

    feature_matrix = np.concatenate([miRNA_feat, drug_feat], axis=0)
    feature_dim = feature_matrix.shape[1]

    drug_df = pd.read_csv(f"{data_dir}/drug.csv")
    mirna_df = pd.read_csv(f"{data_dir}/miRNA.csv")
    drug_name_to_id = {row['drug']: row['id'] for _, row in drug_df.iterrows()}
    mirna_name_to_id = {row['miRNAname']: row['id'] for _, row in mirna_df.iterrows()}

    assoc_df = pd.read_csv(f"{data_dir}/association.csv", header=None, names=['drug', 'mirna'], engine='python')
    positive_pairs = []
    original_associations = []

    for _, row in assoc_df.iterrows():
        drug_name, mirna_name = row['drug'], row['mirna']
        if drug_name in drug_name_to_id and mirna_name in mirna_name_to_id:
            d_id, m_id = drug_name_to_id[drug_name], mirna_name_to_id[mirna_name]
            if 0 <= m_id < miRNA_num and 0 <= d_id < drug_num:
                positive_pairs.append((m_id, d_id + drug_offset))
                original_associations.append((drug_name, mirna_name))

    positive_pairs = np.array(positive_pairs)

    meta_paths = {}

    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for n1, n2 in positive_pairs:
        adj_matrix[n1, n2] = adj_matrix[n2, n1] = 1.0

    # =========================================================
    # 2. Knowledge Graph Integration (NEW)
    # =========================================================
    from knowledge_graph import KnowledgeGraph

    # Prepare entity lists for KG alignment.
    # Preserve the entity order used by the feature tables and identifier maps.
    drug_list_ordered = drug_df['drug'].tolist()
    mirna_list_ordered = mirna_df['miRNAname'].tolist()

    # Construct the knowledge graph with aligned entities.
    kg = KnowledgeGraph(data_dir)
    kg_data = kg.build_graph(drug_list=drug_list_ordered, mirna_list=mirna_list_ordered)

    adj_drug_gene = kg_data['drug_gene'] # (Num_Drugs, Num_Genes)
    adj_mirna_gene = kg_data['mirna_gene'] # (Num_miRNAs, Num_Genes)

    # Construct the full node adjacency matrix.
    # Nodes order: [miRNAs..., Drugs...]
    # Genes are hidden bridge nodes, we don't include them in the main graph node set directly
    # unless we want a full heterogeneous graph.
    # Current architecture assumes only miRNA and Drug nodes in the main graph,
    # and Genes are used for meta-paths.

    meta_paths = {}

    # Convert KG sparse matrices to full node adjacency for meta-path calculation
    # adj_drug_gene is (nD, nG). We need (n_nodes, n_nodes) or intermediate.
    # Actually, we can just do matrix multiplication directly on these smaller matrices
    # and then project result to (miRNA, Drug) space.

    # Let X=[M_MG; M_DG] be the entity--gene incidence matrix.  The
    # entity-space product R = X X^T is used to evaluate higher-order paths
    # without materialising the very large gene--gene matrix X^T X.
    # In particular, the miRNA--drug blocks of successive powers of R are
    # algebraically equivalent to
    #   M_MG M_DG^T, M_MG (X^T X) M_DG^T, and
    #   M_MG (X^T X)^2 M_DG^T,
    # used to obtain distinct semantic path matrices while retaining genes as
    # hidden bridge entities.
    entity_gene = np.concatenate(
        [adj_mirna_gene.toarray(), adj_drug_gene.toarray()], axis=0
    ).astype(np.float64, copy=False)
    entity_semantic = entity_gene @ entity_gene.T

    def add_cross_type_path(name, cross_block):
        """Embed an miRNA-by-drug path matrix in the full entity space."""
        full_path = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        full_path[:miRNA_num, miRNA_num:] = cross_block
        full_path[miRNA_num:, :miRNA_num] = cross_block.T
        meta_paths[name] = full_path

    # l=2: miRNA--Gene--Drug.
    if max_meta_path_length >= 2:
        mp2 = entity_semantic[:miRNA_num, miRNA_num:]
        add_cross_type_path('mp2_mirna_gene_drug', mp2)

    # l=3: miRNA--Gene--Gene--Drug, where the induced Gene--Gene relation
    # is X^T X (genes co-linked by curated miRNA/drug entities).
    if max_meta_path_length >= 3:
        entity_semantic_l3 = entity_semantic @ entity_semantic
        mp3 = entity_semantic_l3[:miRNA_num, miRNA_num:]
        add_cross_type_path('mp3_mirna_gene_gene_drug', mp3)

    # l=4: miRNA--Gene--Gene--Gene--Drug using the second-order induced
    # Gene--Gene relation.  Reuse R^2 above when it is already available.
    if max_meta_path_length >= 4:
        entity_semantic_l4 = entity_semantic_l3 @ entity_semantic
        mp4 = entity_semantic_l4[:miRNA_num, miRNA_num:]
        add_cross_type_path('mp4_mirna_gene_gene_gene_drug', mp4)

    # l=5 and l=6 extend the same induced entity-space walk.
    if max_meta_path_length >= 5:
        entity_semantic_l5 = entity_semantic_l4 @ entity_semantic
        mp5 = entity_semantic_l5[:miRNA_num, miRNA_num:]
        add_cross_type_path('mp5_mirna_gene_gene_gene_gene_drug', mp5)

    if max_meta_path_length >= 6:
        entity_semantic_l6 = entity_semantic_l5 @ entity_semantic
        mp6 = entity_semantic_l6[:miRNA_num, miRNA_num:]
        add_cross_type_path('mp6_mirna_gene_gene_gene_gene_gene_drug', mp6)

    meta_paths['mirna_drug'] = adj_matrix.copy()

    # Row-normalize all meta-path matrices.
    for key in meta_paths:
        row_sums = meta_paths[key].sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        meta_paths[key] = (meta_paths[key] / row_sums).astype(np.float32)

    return feature_matrix, adj_matrix, positive_pairs, meta_paths, drug_df, mirna_df, drug_name_to_id, mirna_name_to_id, original_associations
