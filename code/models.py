"""Shared loss and attention-fusion modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingBCE(nn.Module):
    """Binary cross-entropy with label smoothing."""

    def __init__(self, alpha=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.squeeze()
        target = target.squeeze()
        target = target * (1 - self.alpha) + 0.5 * self.alpha
        loss = F.binary_cross_entropy(pred, target, reduction='none')

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class AttentionFusion(nn.Module):
    """Adaptively fuse a list of node representations."""

    def __init__(self, feat_dim, max_feats=4, dropout=None):
        super().__init__()
        self.attention = nn.Linear(feat_dim, 1)

    def forward(self, feats_list):
        valid_feats = [feat for feat in feats_list if feat.sum() > 0]
        if not valid_feats:
            return feats_list[0]
        if len(valid_feats) == 1:
            return valid_feats[0]

        stacked = torch.stack(valid_feats, dim=1)
        weights = F.softmax(self.attention(stacked), dim=1)
        return (weights * stacked).sum(dim=1)
