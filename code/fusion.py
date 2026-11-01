import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FusionModule(nn.Module):
    """
    Modular Fusion Strategy for combining 128-dim miRNA and Drug features.

    Supported methods:
    - multiply: 128 * 128 -> 128 (Element-wise product)
    - gated: Gated Fusion mechanism z*x + (1-z)*y
    - hybrid: Concat(x*y, x, y) -> Linear -> Output
    - multiply_residual: (x*y) + alpha*(x+y) (Element-wise only)
    """
    def __init__(self, method='multiply', input_dim=128):
        super(FusionModule, self).__init__()
        self.method = method
        self.input_dim = input_dim

        if method == 'concatenate':
            self.output_dim = input_dim * 2
        elif method in ['add', 'average']:
            self.output_dim = input_dim
        elif method == 'multiply':
            self.output_dim = input_dim
        elif method == 'gated':
            # Gated Fusion: Learn a gate z to balance between feat1 and feat2
            # z = sigmoid(W * [feat1, feat2] + b)
            self.gate_net = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.Sigmoid()
            )
            self.output_dim = input_dim
        elif method == 'hybrid':
            # Hybrid Fusion: Combine Multiply interaction with raw features
            # Output = Linear(Concat(feat1 * feat2, feat1, feat2))
            # Preserves both strong interaction (multiply) and individual signals
            self.projection = nn.Sequential(
                nn.Linear(input_dim * 3, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.LeakyReLU(0.2)
            )
            self.output_dim = input_dim
        elif method == 'multiply_residual':
            # Multiply Residual: Element-wise Multiply + Learnable Weighted Sum
            # Output = (feat1 * feat2) + alpha * (feat1 + feat2)
            # Preserves element-wise locality and strong interaction bias
            self.alpha = nn.Parameter(torch.tensor(0.1)) # Initialize with small weight
            self.output_dim = input_dim
        else:
            raise ValueError(f"Unsupported fusion method: {method}")

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: Tensor (batch_size, input_dim) - e.g., Structural features
            feat2: Tensor (batch_size, input_dim) - e.g., Semantic features
        """
        # Ensure inputs are tensors
        if not isinstance(feat1, torch.Tensor):
            feat1 = torch.tensor(feat1, dtype=torch.float32)
        if not isinstance(feat2, torch.Tensor):
            feat2 = torch.tensor(feat2, dtype=torch.float32)

        if self.method == 'concatenate':
            return torch.cat([feat1, feat2], dim=1)
        elif self.method == 'add':
            return feat1 + feat2
        elif self.method == 'average':
            return (feat1 + feat2) / 2
        elif self.method == 'multiply':
            return feat1 * feat2
        elif self.method == 'gated':
            combined = torch.cat([feat1, feat2], dim=1)
            z = self.gate_net(combined)
            # Weighted sum based on the learned gate
            return z * feat1 + (1 - z) * feat2
        elif self.method == 'hybrid':
            interaction = feat1 * feat2
            combined = torch.cat([interaction, feat1, feat2], dim=1)
            return self.projection(combined)
        elif self.method == 'multiply_residual':
            # Element-wise operations only
            return (feat1 * feat2) + self.alpha * (feat1 + feat2)

        return feat1 * feat2

    def get_output_dim(self):
        return self.output_dim

# Static helper for numpy-based pre-processing (if needed outside PyTorch loop)
def numpy_fusion(feat1, feat2, method='multiply'):
    return feat1 * feat2

