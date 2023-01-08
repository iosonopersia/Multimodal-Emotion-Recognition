import torch


# Source: https://arxiv.org/pdf/2105.04906.pdf
def covariance_regularization_term(z):
    # z is a tensor of shape (batch_size, num_features)
    cov_z = torch.cov(z.T)
    cov_z = torch.pow(cov_z, 2)

    d = z.shape[-1]
    cov_z.diagonal()[:] = 0.0
    return cov_z.sum() / d


class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        super(CovarianceLoss, self).__init__()

    def forward(self, za, zp, zn):
        cov_loss = 0.0
        for z in [za, zp, zn]:
            cov_loss += covariance_regularization_term(z)
        return cov_loss
