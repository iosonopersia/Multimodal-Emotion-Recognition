import torch
from torchaudio.pipelines import WAV2VEC2_BASE as WAV2VEC2


class AudioERC(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.is_frozen = False
        self.wav2vec2 = WAV2VEC2.get_model()

        self.classifier_head = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, num_classes),
        )

    def forward(self, input_waveforms, lengths):
        out = self.wav2vec2(
            waveforms=input_waveforms,
            lengths=lengths,
        )
        out = out[0] # out is a tuple of (hidden_states, lengths)
        out = self.classifier_head(out[:, 0, :])
        return out

    def freeze(self):
        if not self.is_frozen:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            self.is_frozen = True

    def unfreeze(self):
        if self.is_frozen:
            for param in self.wav2vec2.parameters():
                param.requires_grad = True
            self.is_frozen = False
