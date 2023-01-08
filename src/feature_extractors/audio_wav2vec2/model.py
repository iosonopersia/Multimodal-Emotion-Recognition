import torch
from torchaudio.pipelines import WAV2VEC2_BASE as WAV2VEC2


class AudioERC(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.is_frozen = False
        self.wav2vec2 = WAV2VEC2.get_model()

        hidden_size = WAV2VEC2._params['encoder_embed_dim']
        self.classifier_head = torch.nn.Sequential(
            # torch.nn.Dropout(0.4),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(), # TODO: Try ReLU
            # torch.nn.Dropout(0.4),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_waveforms, lengths):
        out = self.wav2vec2(
            waveforms=input_waveforms,
            lengths=lengths,
        )
        hidden_states = out[0] # (batch_size, seq_len, hidden_size)
        lengths = out[1] # (batch_size,)

        # Mean pooling over non-padded elements:
        out = torch.cat([torch.mean(hidden_states[[i], :length, :], dim=1) for i, length in enumerate(lengths)], dim=0)

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
