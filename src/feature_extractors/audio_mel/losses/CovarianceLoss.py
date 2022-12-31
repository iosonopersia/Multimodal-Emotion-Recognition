import torch

class CovarianceLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(CovarianceLoss, self).__init__()
        self.eps = eps

    def forward(self, za, zp, zn):
        cov_loss = 0
        for z in [za, zp, zn]:
            cov_loss += self.Lcov(z)
        return cov_loss

    def Lcov(self, z):
        cov_z = torch.cov(z.T)
        # put the diagonal to zero
        cov_z = cov_z - torch.diag(torch.diag(cov_z))
        cov_z = torch.pow(cov_z, 2)
        return cov_z.sum()/ cov_z.shape[-1]




