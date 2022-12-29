import torch
import torch.nn as nn
from torch import Tensor

class AdaptiveTripletMarginLoss(nn.Module):

    def __init__(self, p: float = 2, eps: float = 1e-6, reduction='mean'):
        super(AdaptiveTripletMarginLoss, self).__init__()
        self.p = p
        self.eps = eps
        self.reduction = reduction

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        loss = []
        for i in range(anchor.shape[0]): # Iterate over batch
            loss.append(self.compute_loss(anchor[i], positive[i], negative[i]))

        if self.reduction == 'sum':
            return torch.stack(loss).sum()

        return torch.stack(loss).mean()

    def compute_loss(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        distance_anchor_positive = self.distance(anchor, positive)
        distance_anchor_negative = self.distance(anchor, negative)
        distance_positive_negative = self.distance(positive, negative)

        margin_sim = 1 + 2 / ( torch.exp(4 * distance_anchor_positive) + self.eps)
        margin_dissim = 1 + 2 / ( torch.exp(- (4 * distance_anchor_negative) + 4) + self.eps)
        margin = margin_sim + margin_dissim

        return distance_anchor_positive - (distance_anchor_negative + distance_positive_negative) / 2 + margin

    def distance(self, x1: Tensor, x2: Tensor) -> Tensor:
        return torch.dist(x1, x2, p=self.p)
