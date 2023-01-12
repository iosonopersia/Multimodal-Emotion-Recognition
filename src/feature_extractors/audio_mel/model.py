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
        checkpoint = torch.load("checkpoints/EmoResnet.pth")
        self.resnet18.load_state_dict(checkpoint['model_state_dict'])
        self.append_dropout(self.resnet18)
        self.projector = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, 300),
        )

    def forward(self, x):
        x = self.resnet18(x)
        x = self.projector(x)
        x = normalize(x, p=2, dim=-1)
        return x

    def set_bn_eval(self):
        def _set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.resnet18.apply(_set_bn_eval)


    def append_dropout(self, model, rate=0.2):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.append_dropout(module)
            if isinstance(module, nn.ReLU):
                new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=True))
                setattr(model, name, new)

# add to AudioMelFeatureExtractor class a classification head for emotion (output 7 classes)
class TestAudioExtractor(nn.Module):
    def __init__(self, load_checkpoint=True):
        super(TestAudioExtractor, self).__init__()
        self.audio_extractor = AudioMelFeatureExtractor()
        self.audio_extractor.eval()
        if load_checkpoint:
            checkpoint = torch.load("checkpoints/audio_mel/checkpoint.pth")
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
        with torch.inference_mode():
            x = self.audio_extractor(x)
        x = self.classifier(x)
        return x


class EmoResnet (nn.Module):
    def __init__(self):
        super(EmoResnet, self).__init__()
        self.resnet18 = resnet18(weights="DEFAULT")
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 7)
            # nn.ReLU(),
            # nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.resnet18(x)
        x = self.classifier(x)
        return x
