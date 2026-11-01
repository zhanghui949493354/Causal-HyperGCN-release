import torch
import torch.nn as nn
import torch.nn.functional as F
from models import LabelSmoothingBCE, AttentionFusion

class ImprovedProjector(nn.Module):
    """
    Dimensionality Reduction Module (256 -> 32).
    Can be a simple MLP or an AutoEncoder style reduction.
    """
    def __init__(self, input_dim=256, hidden_dim=64, output_dim=32, dropout=0.5, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2) # Activation on output to match GNN input expectation
        )

        # Optional direct projection for residual connection if dims match (unlikely here 256->32)
        if input_dim != output_dim and use_residual:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.layer1(x)
        out = self.layer2(h)

        if self.use_residual:
            res = self.shortcut(x)
            out = out + res

        return out

class GATLayer(nn.Module):
    """Simple GAT Layer"""
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        # Broadcast logic for attention (simplification for dense adj)
        # Dense adjacency is used at the current graph scale.
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class HypergraphConv(nn.Module):
    """
    Hypergraph Convolution Layer
    X' = sigma( D_V^-0.5 * H * W * D_E^-1 * H^T * D_V^-0.5 * X * Theta )
    """
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()
        self.theta = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, H):
        """
        x: (N, in_features)
        H: (N, M) Incidence matrix where M is number of hyperedges (clusters)
        """
        # 1. Node -> Hyperedge (Aggregation)
        # H^T * X: (M, N) * (N, D) -> (M, D)
        # We need to normalize by Degree of Hyperedge (D_E)
        # D_E is sum of H over rows (number of nodes in hyperedge)
        d_e = H.sum(dim=0) # (M,)
        d_e_inv = torch.pow(d_e + 1e-8, -1.0) # (M,)
        d_e_inv_mat = torch.diag(d_e_inv) # (M, M)

        # H^T * X
        hyperedge_feat = torch.matmul(H.t(), x) # (M, D)
        hyperedge_feat = torch.matmul(d_e_inv_mat, hyperedge_feat) # (M, D) * (M, M) -> (M, D)

        # 2. Hyperedge -> Node (Distribution)
        # H * Y: (N, M) * (M, D) -> (N, D)
        # Normalize by Degree of Node (D_V)
        # D_V is sum of H over cols (number of hyperedges a node belongs to)
        # In KMeans, each node belongs to exactly 1 cluster, so D_V is all 1s.
        # But if we use soft clustering or multiple clusterings, it matters.
        d_v = H.sum(dim=1) # (N,)
        d_v_inv_sqrt = torch.pow(d_v + 1e-8, -0.5) # (N,)
        d_v_inv_sqrt_mat = torch.diag(d_v_inv_sqrt) # (N, N)

        # Pre-normalization (D_V^-0.5)
        # Since we do D_V^-0.5 * H * ... * H^T * D_V^-0.5
        # We can apply one D_V^-0.5 at the end, or symmetrically.
        # Let's do symmetric.

        # Re-calculate flow:
        # X_norm = D_V^-0.5 * X
        x_norm = torch.matmul(d_v_inv_sqrt_mat, x)

        # Node -> Hyperedge
        he_feat = torch.matmul(H.t(), x_norm)
        he_feat = torch.matmul(d_e_inv_mat, he_feat)

        # Hyperedge -> Node
        node_feat = torch.matmul(H, he_feat)

        # Post-normalization
        node_feat = torch.matmul(d_v_inv_sqrt_mat, node_feat)

        # 3. Transformation (Theta)
        out = self.theta(node_feat)
        out = self.bn(out)
        out = self.act(out)
        out = F.dropout(out, self.dropout, training=self.training)

        return out

class CausalHyperGCN(nn.Module):
    """
    Causal Enhanced Dual-Stream Hypergraph GCN.
    Stream 1: Standard GCN (Local Structure)
    Stream 2: Hypergraph GCN (Global Group Structure via Clustering)
    Fusion: Weighted Sum or Concat
    Causal: Environmental Prototypes Injection
    """
    def __init__(self, input_dim=32, hidden_dims=[32, 32], dropout=0.5, num_prototypes=10, use_local_stream=True, use_global_stream=True):
        super().__init__()
        self.dropout = dropout
        self.num_prototypes = num_prototypes
        self.hidden_dims = hidden_dims
        self.use_local_stream = use_local_stream
        self.use_global_stream = use_global_stream

        if not use_local_stream and not use_global_stream:
            raise ValueError("At least one stream (local or global) must be enabled in CausalHyperGCN.")

        # --- Stream 1: Local GCN ---
        if self.use_local_stream:
            self.gcn_layers = nn.ModuleList()
            prev_dim = input_dim
            for h_dim in hidden_dims:
                self.gcn_layers.append(
                    nn.Sequential(
                        nn.Linear(prev_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(dropout)
                    )
                )
                prev_dim = h_dim

        # --- Stream 2: Hypergraph GCN ---
        if self.use_global_stream:
            self.hgnn_layers = nn.ModuleList()
            prev_dim_h = input_dim
            for h_dim in hidden_dims:
                self.hgnn_layers.append(
                    HypergraphConv(prev_dim_h, h_dim, dropout)
                )
                prev_dim_h = h_dim

        # --- Fusion ---
        # Use the last hidden dim as the feature dim for fusion
        last_hidden_dim = hidden_dims[-1] if hidden_dims else input_dim
        if self.use_local_stream and self.use_global_stream:
            self.fusion_module = AttentionFusion(feat_dim=last_hidden_dim, dropout=dropout)

        self.output_dim = last_hidden_dim

        # --- Causal Module ---
        if self.num_prototypes > 0:
            self.context_prototypes = nn.Parameter(torch.zeros(num_prototypes, input_dim))
            nn.init.xavier_uniform_(self.context_prototypes.data)
            self.w_att = nn.Linear(input_dim, input_dim)
        else:
            self.context_prototypes = None
            self.w_att = None

    def forward(self, x, adj, H, causal_inference=False):
        """
        x: (N, D)
        adj: (N, N) Adjacency
        H: (N, M) Hypergraph Incidence Matrix
        """
        p_z_x = None # Default value

        # --- Causal P(z|X) and Context Injection ---
        if self.num_prototypes > 0 and self.w_att is not None:
            x_att = torch.tanh(self.w_att(x))
            att_scores = torch.matmul(x_att, self.context_prototypes.t())
            p_z_x = F.softmax(att_scores, dim=1)

            context = torch.matmul(p_z_x, self.context_prototypes)
            if causal_inference:
                avg_context = torch.mean(self.context_prototypes, dim=0, keepdim=True)
                context = avg_context.expand(x.size(0), -1)

            if context.shape[1] == x.shape[1]:
                x_context = x + context
            else:
                x_context = x
        else: # No causal module
            x_context = x

        # --- Stream Processing ---
        out_gcn = None
        if self.use_local_stream:
            out_gcn = x_context
            for layer in self.gcn_layers:
                support = torch.matmul(adj, out_gcn)
                out_gcn = layer(support)

        out_hgnn = None
        if self.use_global_stream:
            out_hgnn = x_context
            for layer in self.hgnn_layers:
                out_hgnn = layer(out_hgnn, H)

        # --- Fusion ---
        if self.use_local_stream and self.use_global_stream:
            out = self.fusion_module([out_gcn, out_hgnn])
        elif self.use_local_stream:
            out = out_gcn
        elif self.use_global_stream:
            out = out_hgnn
        else:
            out = torch.zeros(x.size(0), self.output_dim).to(x.device)

        return out, p_z_x

class ImprovedGNN(nn.Module):
    """
    Improved GNN Module.
    Supports 'gcn', 'gat', and various causal ablations.
    """
    def __init__(self, input_dim=32, hidden_dims=[32, 32], dropout=0.5, gnn_type='gcn', use_virtual_node=False, num_prototypes=15):
        super().__init__()
        self.gnn_type = gnn_type
        self.use_virtual_node = use_virtual_node

        # Configure the local and global streams for each model variant.
        if gnn_type in ['causal_hyper', 'causal_gcn', 'causal_hyper_global_only']:
            use_local = gnn_type in ['causal_hyper', 'causal_gcn']
            use_global = gnn_type in ['causal_hyper', 'causal_hyper_global_only']

            self.impl = CausalHyperGCN(
                input_dim,
                hidden_dims,
                dropout,
                num_prototypes=num_prototypes,
                use_local_stream=use_local,
                use_global_stream=use_global
            )
            self.output_dim = self.impl.output_dim
            return

        # Standard GCN/GAT variants.
        self.dropout = dropout
        self.layers = nn.ModuleList()

        # Virtual Node Initialization
        if self.use_virtual_node:
            self.virtual_node_emb = nn.Parameter(torch.zeros(1, input_dim))
            nn.init.xavier_uniform_(self.virtual_node_emb.data)
            self.vn_mlps = nn.ModuleList()

        prev_dim = input_dim

        if gnn_type == 'gat':
            # GAT Layers
            for i, h_dim in enumerate(hidden_dims):
                if self.use_virtual_node:
                    self.vn_mlps.append(nn.Sequential(
                        nn.Linear(h_dim * 2, h_dim * 2), nn.LayerNorm(h_dim * 2), nn.LeakyReLU(0.2), nn.Dropout(dropout),
                        nn.Linear(h_dim * 2, h_dim), nn.LayerNorm(h_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout)
                    ))
                self.layers.append(
                    GATLayer(in_features=prev_dim, out_features=h_dim, dropout=dropout, alpha=0.2, concat=True)
                )
                prev_dim = h_dim
        else:
            # Original simplified GCN
            for i, h_dim in enumerate(hidden_dims):
                if self.use_virtual_node:
                    self.vn_mlps.append(nn.Sequential(
                        nn.Linear(h_dim * 2, h_dim * 2), nn.LayerNorm(h_dim * 2), nn.LeakyReLU(0.2), nn.Dropout(dropout),
                        nn.Linear(h_dim * 2, h_dim), nn.LayerNorm(h_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout)
                    ))
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(prev_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(dropout)
                    )
                )
                prev_dim = h_dim

        self.output_dim = prev_dim

    def forward(self, x, adj, H=None, causal_inference=False):
        if hasattr(self, 'impl'):
            if self.impl.use_global_stream and H is None:
                print("Warning: Hypergraph matrix H not provided for a global-stream model. Using a zero matrix.")
                H = torch.zeros(x.size(0), 1).to(x.device)
            return self.impl(x, adj, H, causal_inference)

        # --- Standard GCN/GAT forward pass ---
        out = x
        if self.use_virtual_node:
            vx = self.virtual_node_emb

        if self.gnn_type == 'gat':
            for i, layer in enumerate(self.layers):
                h_local = layer(out, adj)
                if self.use_virtual_node:
                    pooled_mean = torch.mean(h_local, dim=0, keepdim=True)
                    pooled_max, _ = torch.max(h_local, dim=0, keepdim=True)
                    pooled = torch.cat([pooled_mean, pooled_max], dim=1)
                    vx_update = self.vn_mlps[i](pooled)
                    vx = vx + vx_update
                    out = h_local + vx
                else:
                    out = h_local
                if i < len(self.layers) - 1:
                    out = F.dropout(out, self.dropout, training=self.training)
        else: # GCN
            for i, layer in enumerate(self.layers):
                support = torch.matmul(adj, out)
                h_local = layer(support)
                if self.use_virtual_node:
                    pooled_mean = torch.mean(h_local, dim=0, keepdim=True)
                    pooled_max, _ = torch.max(h_local, dim=0, keepdim=True)
                    pooled = torch.cat([pooled_mean, pooled_max], dim=1)
                    vx_update = self.vn_mlps[i](pooled)
                    vx = vx + vx_update
                    out = h_local + vx
                else:
                    out = h_local
                if i < len(self.layers) - 1:
                    out = F.dropout(out, self.dropout, training=self.training)
        return out

class HGGNGAT(nn.Module):
    """
    HGGN-style GAT with multiple virtual nodes.
    Constructs an augmented graph with 10% virtual nodes fully connected to real nodes.
    """
    def __init__(self, input_dim=32, hidden_dims=[32, 32], dropout=0.5, num_nodes=None):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        # Use virtual nodes equal to 10% of the graph size.
        self.num_real_nodes = num_nodes if num_nodes else 572
        self.num_virtual_nodes = int(0.1 * self.num_real_nodes)

        # Virtual nodes embedding (Learnable)
        self.virtual_node_emb = nn.Parameter(torch.empty(self.num_virtual_nodes, input_dim))
        nn.init.xavier_uniform_(self.virtual_node_emb.data)

        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(
                GATLayer(in_features=prev_dim, out_features=h_dim, dropout=dropout, alpha=0.2, concat=True)
            )
            prev_dim = h_dim

        self.output_dim = prev_dim

    def forward(self, x, adj):
        """
        x: (N, D) - Real node features
        adj: (N, N) - Real node adjacency
        """
        device = x.device
        N = x.size(0)
        D = x.size(1)
        Nv = self.num_virtual_nodes

        # 1. Augment Features
        # X_new = [X_real; X_virtual]
        x_aug = torch.cat([x, self.virtual_node_emb], dim=0) # (N+Nv, D)

        # 2. Augment Adjacency Matrix
        # A_new = [[A, M], [M.T, I]]
        # M is (N, Nv) full of 1s
        # I is (Nv, Nv) identity

        # Construct M
        M = torch.ones(N, Nv, device=device)

        # Construct I
        I = torch.eye(Nv, device=device)

        # Build rows
        top_row = torch.cat([adj, M], dim=1) # (N, N+Nv)
        bottom_row = torch.cat([M.t(), I], dim=1) # (Nv, N+Nv)

        # Full Adj
        adj_aug = torch.cat([top_row, bottom_row], dim=0) # (N+Nv, N+Nv)

        # 3. Message Passing on Augmented Graph
        out = x_aug
        for layer in self.layers:
            out = layer(out, adj_aug)

        # 4. Extract Real Node Features
        out_real = out[:N, :]

        return out_real
