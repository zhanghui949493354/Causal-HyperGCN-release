import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SemanticAttention(nn.Module):
    """Fuse meta-path embeddings with HAN-style semantic attention."""
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # z: [batch_size, num_paths, feature_dim]
        w = self.project(z)  # [batch_size, num_paths, 1]
        beta = torch.softmax(w, dim=1)  # [batch_size, num_paths, 1]
        return (beta * z).sum(1)  # [batch_size, feature_dim]

class ImprovedMetaPathEncoder(nn.Module):
    """Encode each available meta-path and fuse the results by attention."""
    def __init__(self, input_dim=32, hidden_dim=32, dropout=0.5, use_attention=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_attention = use_attention

        # Transformations are created for the paths supplied by the data loader.
        self.path_transforms = nn.ModuleDict()

        if use_attention:
            self.semantic_attention = SemanticAttention(hidden_dim)

        self.bn = nn.BatchNorm1d(hidden_dim)

    def _get_transform(self, path_name):
        if path_name not in self.path_transforms:
            self.path_transforms[path_name] = nn.Linear(self.input_dim, self.hidden_dim).to(next(self.parameters()).device)
        return self.path_transforms[path_name]

    def forward(self, x, meta_paths_dict):
        """
        Args:
            x: Node features [N, D]
            meta_paths_dict: Dict {path_name: adj_matrix}
        """
        embeddings = []

        for name, adj in meta_paths_dict.items():
            if adj.sum() == 0:
                continue

            transform = self._get_transform(name)
            h = transform(x)
            h = torch.matmul(adj, h)
            h = self.bn(h)
            h = F.leaky_relu(h, 0.2)
            h = F.dropout(h, self.dropout, training=self.training)
            embeddings.append(h)

        if not embeddings:
            return torch.zeros(x.size(0), self.hidden_dim).to(x.device)

        z = torch.stack(embeddings, dim=1)

        if self.use_attention:
            out = self.semantic_attention(z)
        else:
            out = z.mean(dim=1)

        return out
