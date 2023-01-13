import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.nn.functional import normalize

'''here we use ResNet18 as the backbone of the encoder network while the projector con-
sists a linear fully connected layer which project the em-
bedding of encoder network to desired representations'''

class AudioMelFeatureExtractor(nn.Module):
    def __init__(self):
        super(AudioMelFeatureExtractor, self).__init__()
        self.resnet18 = resnet18(weights=None)
        self.projector = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, 300),
        )

    def forward(self, x):
        x = self.resnet18(x)
        x = self.projector(x)
        x = normalize(x, p=2, dim=-1)
        return x
