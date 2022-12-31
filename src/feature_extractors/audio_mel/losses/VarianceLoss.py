import torch
import torch.nn.functional as F

class VarianceLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(VarianceLoss, self).__init__()
        self.eps = eps

    def forward(self, za, zp, zn):
        var_loss = 0
        for z in [za, zp, zn]:
            var_loss += self.Lvar(z)
        return var_loss

    def Lvar(self, z):

        var_z = torch.var(z, dim=0)
        result = F.relu((1 - torch.sqrt(var_z + self.eps))).mean()
        return result

