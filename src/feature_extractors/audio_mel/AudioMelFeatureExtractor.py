import torch
import torch.nn as nn
from torchvision.models import resnet18

'''here we use ResNet18 as the backbone of the encoder network while the projector con-
sists a linear fully connected layer which project the em-
bedding of encoder network to desired representations'''

class AudioMelFeatureExtractor(nn.Module):
    def __init__(self):
        super(AudioMelFeatureExtractor, self).__init__()
        self.resnet18 = resnet18(weights=None)
        # self.resnet18.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(1000, 768),
            nn.ReLU(),
            nn.Linear(768, 300)
        )
    def forward(self, x):
        x = self.resnet18(x)
        x = self.projector(x)
        return x

# add to AudioMelFeatureExtractor class a classification head for emotion (output 7 classes)
class TestAudioExtractor(nn.Module):
    def __init__(self, load_checkpoint=True):
        super(TestAudioExtractor, self).__init__()
        self.audio_extractor = AudioMelFeatureExtractor()
        self.audio_extractor.eval()
        if load_checkpoint:
            checkpoint = torch.load("checkpoints/audio_feature_extractor.pth")
            self.audio_extractor.load_state_dict(checkpoint['model_state_dict'])
        for param in self.audio_extractor.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        x = self.audio_extractor(x)
        x = self.classifier(x)
        return x


