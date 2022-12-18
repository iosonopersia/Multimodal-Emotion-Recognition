import torchaudio
import torch
from transformers import RobertaModel
from utils import apply_padding

class MeanPoolingWithoutPadding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, lengths):
        # features: (batch_size, seq_len, embedding_size)
        pooled_features = torch.cat([features[[i], :lengths[i], :].mean(dim=1) for i in range(features.shape[0])], dim=0)
        # pooled_features: (batch_size, embedding_size)
        return pooled_features


class MeanPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        # features: (batch_size, seq_len, embedding_size)
        pooled_features = features.mean(dim=1)
        # pooled_features: (batch_size, embedding_size)
        return pooled_features


class FeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base').to(device)
        self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec = self.bundle.get_model().to(device)
        self.mean_pooling = MeanPoolingWithoutPadding().to(device)
        self.device = device

    def forward(self, batch_text, batch_audio):
        with torch.inference_mode():
            batch_text_features = []
            batch_audio_features = []
            for text, audio in zip(batch_text, batch_audio):
                # text embeddings
                text_input_ids = text["input_ids"]
                text_attention_mask = text["attention_mask"]

                text_features = self.roberta(input_ids=text_input_ids.to(self.device), attention_mask=text_attention_mask.to(self.device))
                text_features = text_features.last_hidden_state
                text_features = self.mean_pooling(text_features, text_attention_mask.sum(dim=1))

                # audio embeddings

                # Unfortunately, wav2vec2 requires a lot of memory, so we cannot afford to process all the batch at once:
                audio_features = []
                for a in audio:
                    _audio_features, _ = self.wav2vec.extract_features(waveforms=a.to(self.device))
                    _audio_features = _audio_features[-1] # last hidden state
                    audio_features.append(_audio_features)

                # ! Applying padding here is useless if we use MeanPoolingWithoutPadding!
                audio_features, lengths = apply_padding(audio_features)
                audio_features = self.mean_pooling(audio_features, lengths)

                # Add batch dimension
                text_features = text_features.unsqueeze(dim=0)
                audio_features = audio_features.unsqueeze(dim=0)

                batch_text_features.append(text_features)
                batch_audio_features.append(audio_features)

            batch_text_features, batch_text_features_length = apply_padding(batch_text_features)
            batch_text_features = {"text": batch_text_features.cpu(), "lengths": batch_text_features_length.cpu()}

            batch_audio_features, batch_audio_features_length = apply_padding(batch_audio_features)
            batch_audio_features = {"audio": batch_audio_features.cpu(), "lengths": batch_audio_features_length.cpu()}

        return batch_text_features, batch_audio_features
