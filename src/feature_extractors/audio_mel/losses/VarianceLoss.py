import torch
import torch.nn.functional as F


# Source: https://arxiv.org/pdf/2105.04906.pdf
def variance_regularization_term(z, gamma=1.0, eps=1e-6):
    # z is a tensor of shape (batch_size, num_features)
    var_z = torch.var(z, dim=0)

    d = z.shape[-1]
    var_z = F.relu(gamma - torch.sqrt(var_z + eps))
    return torch.sum(var_z) / d


class VarianceLoss(torch.nn.Module):
    def __init__(self, gamma=1.0, eps=1e-6):
        super(VarianceLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, za, zp, zn):
        var_loss = 0.0
        for z in [za, zp, zn]:
            var_loss += variance_regularization_term(z, gamma=self.gamma, eps=self.eps)
        return var_loss
