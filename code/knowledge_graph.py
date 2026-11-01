import pandas as pd
import numpy as np
import scipy.sparse as sp
import os

class KnowledgeGraph:
    """
    Biomedical Knowledge Graph (BioKG) Construction Module.

    This module formalizes the construction of a Knowledge Graph from heterogeneous data sources.
    Instead of treating data as simple matrices, we define:
    - Entities: Drugs, miRNAs, Genes
    - Relations:
        - (Drug, targets, Gene)
        - (miRNA, regulates, Gene)
        - (miRNA, associated_with, Drug) [Target to predict]
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entities = {'drug': {}, 'mirna': {}, 'gene': {}}
        self.relations = []

        # Mappings
        self.gene_to_id = {}
        self.id_to_gene = {}

    def build_graph(self, drug_list=None, mirna_list=None):
        """
        Constructs the KG triples from raw CSV files.
        Args:
            drug_list (list): List of drug names to enforce order and dimension.
            mirna_list (list): List of miRNA names to enforce order and dimension.
        """
        # 1. Load Data
        drug_gene_path = os.path.join(self.data_dir, 'drug_gene_matrix.csv')
        mirna_gene_path = os.path.join(self.data_dir, 'miRNA_gene_matrix.csv')

        if not os.path.exists(drug_gene_path) or not os.path.exists(mirna_gene_path):
            raise FileNotFoundError("KG source files missing!")

        # Read CSVs (Rows: Entities, Cols: Genes)
        df_drug_gene = pd.read_csv(drug_gene_path, index_col=0)
        df_mirna_gene = pd.read_csv(mirna_gene_path, index_col=0)

        # 2. Unify Gene Entities
        all_genes = df_drug_gene.columns.tolist()
        common_genes = [g for g in all_genes if g in df_mirna_gene.columns]


        # 3. Align Entities (Crucial Step!)
        # We must ensure the rows of the matrix correspond exactly to drug_list and mirna_list

        # --- Align Drugs ---
        if drug_list is not None:
            aligned_drug_mat = np.zeros((len(drug_list), len(common_genes)), dtype=np.float32)
            # Create a lookup for source rows
            source_drug_indices = {name: i for i, name in enumerate(df_drug_gene.index)}

            match_count = 0
            for i, drug in enumerate(drug_list):
                if drug in source_drug_indices:
                    # Copy row
                    row_idx = source_drug_indices[drug]
                    aligned_drug_mat[i] = df_drug_gene.iloc[row_idx][common_genes].values
                    match_count += 1
            drug_gene_mat = aligned_drug_mat
        else:
            drug_gene_mat = df_drug_gene[common_genes].values

        # --- Align miRNAs ---
        if mirna_list is not None:
            aligned_mirna_mat = np.zeros((len(mirna_list), len(common_genes)), dtype=np.float32)
            source_mirna_indices = {name: i for i, name in enumerate(df_mirna_gene.index)}

            match_count = 0
            for i, mirna in enumerate(mirna_list):
                if mirna in source_mirna_indices:
                    row_idx = source_mirna_indices[mirna]
                    aligned_mirna_mat[i] = df_mirna_gene.iloc[row_idx][common_genes].values
                    match_count += 1
            mirna_gene_mat = aligned_mirna_mat
        else:
            mirna_gene_mat = df_mirna_gene[common_genes].values

        # 4. Construct Relations (Edges)
        drug_gene_mat = (drug_gene_mat > 0).astype(np.float32)
        adj_drug_gene = sp.csr_matrix(drug_gene_mat)

        mirna_gene_mat = (mirna_gene_mat > 0).astype(np.float32)
        adj_mirna_gene = sp.csr_matrix(mirna_gene_mat)

        return {
            'drug_gene': adj_drug_gene,
            'mirna_gene': adj_mirna_gene,
            'gene_list': common_genes
        }

if __name__ == "__main__":
    # Test
    kg = KnowledgeGraph('./data')
    kg.build_graph()
