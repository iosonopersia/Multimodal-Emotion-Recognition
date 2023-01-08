import torch
from losses.AdaptiveTripletMarginLoss import AdaptiveTripletMarginLoss
from losses.VarianceLoss import VarianceLoss
from losses.CovarianceLoss import CovarianceLoss


# Source: https://arxiv.org/pdf/2206.02187v1.pdf
class M2FNetAudioEmbeddingLoss(torch.nn.Module):
    def __init__(self, adaptive=True, covariance_enabled=True, variance_enabled=True):
        super(M2FNetAudioEmbeddingLoss, self).__init__()
        self.adaptive = adaptive
        self.covariance_enabled = covariance_enabled
        self.variance_enabled = variance_enabled

        if self.adaptive:
            self.triplet_loss = AdaptiveTripletMarginLoss()
        else:
            self.triplet_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2)
        self.covariance_loss = CovarianceLoss()
        self.variance_loss = VarianceLoss()

    def forward(self, anchor, positive, negative):
        loss =  20.0 * self.triplet_loss(anchor, positive, negative)
        if self.covariance_enabled:
            loss += 5.0 * self.covariance_loss(anchor, positive, negative)
        if self.variance_enabled:
            loss += 1.0 * self.variance_loss(anchor, positive, negative)

        return loss
