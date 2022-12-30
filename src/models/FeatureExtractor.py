import torchaudio
import torch
from transformers import RobertaModel
from utils import apply_padding

#TODO check if this is correct
class MeanPoolingWithoutPadding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, lengths):
        # features: (batch_size, seq_len, embedding_size)
        pooled_features = torch.cat([features[[i], :lengths[i], :].mean(dim=1) for i in range(features.shape[0])], dim=0)
        # pooled_features: (batch_size, embedding_size)
        return pooled_features

class FeatureExtractor(torch.nn.Module):
    def __init__(self, roberta_checkpoint='roberta-base'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_checkpoint, add_pooling_layer=False)
        self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec = self.bundle.get_model()
        self.mean_pooling = MeanPoolingWithoutPadding()

    def forward(self, batch_text, batch_audio):
        batch_text_features = []
        batch_audio_features = []
        for text, audio in zip(batch_text, batch_audio):
            # text embeddings
            text_input_ids = text["input_ids"]
            text_attention_mask = text["attention_mask"]

            text_features = self.roberta(input_ids=text_input_ids, attention_mask=text_attention_mask)
            text_features = text_features.last_hidden_state[:, 0, :]

            # audio embeddings
            # Unfortunately, wav2vec2 requires a lot of memory, so we cannot afford to process all the batch at once:
            # TODO find a way to process the whole batch at once
            audio_features = []
            for a in audio:
                _audio_features, _ = self.wav2vec.extract_features(waveforms=a)
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

        batch_text_features, dialogue_lengths = apply_padding(batch_text_features)
        batch_audio_features, _ = apply_padding(batch_audio_features)

        # Prepare padding masks:
        padding_mask = torch.ones(batch_text_features.shape[0], batch_text_features.shape[1],
                                    dtype=torch.bool, device=batch_text_features.device)
        for i in range(batch_text_features.shape[0]):
            padding_mask[i, :dialogue_lengths[i]] = False

        return batch_text_features, batch_audio_features, padding_mask
