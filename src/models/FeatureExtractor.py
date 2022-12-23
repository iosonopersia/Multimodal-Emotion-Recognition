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
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("tae898/emoberta-base")
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

            # TODO pass the previous and next utterance to the feature extractor
            text_features = self.roberta(input_ids=text_input_ids, attention_mask=text_attention_mask)
            text_features = text_features.last_hidden_state
            text_features = text_features[:, 0, :]  # take the CLS token
            # text_features = self.mean_pooling(text_features, text_attention_mask.sum(dim=1))

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

        # Lengths to attention masks:
        attention_mask = torch.ones(batch_text_features.shape[0], batch_text_features.shape[1],
                                    dtype=torch.bool, device=batch_text_features.device)
        for i in range(batch_text_features.shape[0]):
            attention_mask[i, :dialogue_lengths[i]] = False

        return batch_text_features, batch_audio_features, attention_mask



# class text_extractor (torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.roberta = RobertaModel.from_pretrained('roberta-base')

#     def forward(self, batch_text, padding_mask):

#         x = self.roberta(batch_text, padding_mask)
#         x = x.last_hidden_state
#         cls_embedding = x[0]

#         return cls_embedding

# class text_emotion_classifier(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.roberta = text_extractor()
#         self.fc = torch.nn.Sequential(
#         torch.nn.Linear(768, 384),
#         torch.nn.ReLU(),
#         torch.nn.Linear(384, 7))

#     def forward(self, batch_text, padding_mask):

#         cls_embedding = self.roberta(batch_text, padding_mask)
#         x = self.fc(cls_embedding)

#         return x


