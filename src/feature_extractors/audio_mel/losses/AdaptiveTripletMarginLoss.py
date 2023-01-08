import torch
from torch import Tensor


# Source: https://arxiv.org/pdf/2206.02187v1.pdf
class AdaptiveTripletMarginLoss(torch.nn.Module):
    def __init__(self, p: float = 2, eps: float = 1e-6, reduction: str = 'mean'):
        super(AdaptiveTripletMarginLoss, self).__init__()
        self.p = p
        self.eps = eps
        self.reduction = reduction
        if self.reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        loss = self.compute_loss(anchor, positive, negative)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def compute_loss(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        '''
        anchor: Tensor of shape (B, embedding_size) where B is the batch size
        positive: Tensor of shape (B, embedding_size) where B is the batch size
        negative: Tensor of shape (B, embedding_size) where B is the batch size
        '''
        distance_anchor_positive = self.distance(anchor, positive)
        distance_anchor_negative = self.distance(anchor, negative)
        distance_positive_negative = self.distance(positive, negative)
        margin = self.adaptive_margin(distance_anchor_positive, distance_anchor_negative)

        return distance_anchor_positive - (torch.add(distance_anchor_negative, distance_positive_negative) / 2.0) + margin

    def adaptive_margin(self, distance_anchor_positive: Tensor, distance_anchor_negative: Tensor) -> Tensor:
        margin_sim = 1.0 + 2.0 / (torch.exp(4.0 * distance_anchor_positive) + self.eps)
        margin_dissim = 1.0 + 2.0 / (torch.exp(-4.0 * distance_anchor_negative + 4.0) + self.eps)

        return torch.add(margin_sim, margin_dissim)

    def distance(self, x1: Tensor, x2: Tensor) -> Tensor:
        return torch.norm(x1 - x2, p=self.p, dim=-1)
