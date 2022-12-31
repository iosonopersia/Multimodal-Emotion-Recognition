import torch
import torch.nn as nn
from torch import Tensor

class AdaptiveTripletMarginLoss(nn.Module):

    def __init__(self, p: float = 2, eps: float = 1e-6):
        super(AdaptiveTripletMarginLoss, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:

        return self.compute_loss(anchor, positive, negative).mean()


    def compute_loss(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        '''
        anchor: Tensor of shape (B, embedding_size) where B is the batch size
        positive: Tensor of shape (B, embedding_size) where B is the batch size
        negative: Tensor of shape (B, embedding_size) where B is the batch size
        '''
        distance_anchor_positive = self.distance(anchor, positive)
        distance_anchor_negative = self.distance(anchor, negative)
        distance_positive_negative = self.distance(positive, negative)

        margin_sim = 1 + (2 / ( torch.exp(4 * distance_anchor_positive) + self.eps))
        margin_dissim = 1 + (2 / ( torch.exp((-4 * distance_anchor_negative) + 4) + self.eps))

        margin = torch.add(margin_sim, margin_dissim)

        return distance_anchor_positive - (distance_anchor_negative + distance_positive_negative) / 2 + margin

    def distance(self, x1: Tensor, x2: Tensor) -> Tensor:
        return torch.norm(x1 - x2, p=self.p, dim=1)
