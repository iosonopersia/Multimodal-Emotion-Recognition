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
        hidden_states = out[0] # (batch_size, seq_len, 768)
        lengths = out[1] # (batch_size,)

        # Mean pooling over non-padded elements:
        for i, length in enumerate(lengths):
            hidden_states[i, length:, :] = 0
        out = torch.sum(hidden_states, dim=1) / lengths.unsqueeze(dim=1).type(torch.float32)

        out = self.classifier_head(out)
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
