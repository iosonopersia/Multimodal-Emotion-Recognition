import torch
import torch.nn as nn


class FusionAttention(nn.Module):
    def __init__(self, embedding_size, n_head):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embedding_size, n_head, batch_first=True)
        self.linear = nn.Linear(2*embedding_size, embedding_size)
        self.relu = nn.ReLU()


    def forward(self, embedding_text, embedding_audio):
        x, _ = self.multihead_attention(embedding_text, embedding_audio, embedding_text)
        x = torch.cat((x, embedding_text), dim=2)
        x = self.relu(x)
        x = self.linear(x)
        return x

class M2FNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model_audio = config.AUDIO.EMBEDDING_SIZE
        d_model_text = config.TEXT.EMBEDDING_SIZE
        n_head_audio = config.AUDIO.N_HEAD
        n_head_text = config.TEXT.N_HEAD
        n_layers_audio = config.AUDIO.N_LAYERS
        n_layers_text = config.TEXT.N_LAYERS
        n_fam_layers = config.FAM.N_LAYERS
        n_layers_classifier = config.CLASSIFIER.N_LAYERS
        hidden_size_classifier = config.CLASSIFIER.HIDDEN_SIZE
        output_size = config.OUTPUT_SIZE

        #Audio and text encoders [Dialogue-level]
        audio_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_audio, nhead=n_head_audio)
        self.audio_transformer = nn.TransformerEncoder(audio_encoder_layer, num_layers=n_layers_audio)

        text_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_text, nhead=n_head_text)
        self.text_transformer = nn.TransformerEncoder(text_encoder_layer, num_layers=n_layers_text)

        # Fusion layer [Dialogue-level]
        self.fusion_layers = nn.ModuleList([
            FusionAttention(embedding_size=d_model_audio, n_head=n_head_audio)
        ] * n_fam_layers)

        # Output layer [Dialogue-level]
        classifier_head = [nn.Linear(d_model_text + d_model_audio, hidden_size_classifier)]
        for _ in range(n_layers_classifier - 2):
            classifier_head.append(nn.ReLU())
            classifier_head.append(nn.Linear(hidden_size_classifier, hidden_size_classifier))

        classifier_head.append(nn.ReLU())
        classifier_head.append(nn.Linear(hidden_size_classifier, output_size))

        self.output_layer = nn.Sequential(*classifier_head)

    def forward(self, text, audio):
        text, text_lengths = text["text"], text["lengths"]
        audio, audio_lengths = audio["audio"], audio["lengths"]

        # Audio and text encoders
        # TODO pass the previous and next utterance to the feature extractor
        # TODO construct the mask for the audio and text encoders (starting from the lengths)
        dialogue_audio_features = self.audio_transformer(audio, mask=None)
        dialogue_text_features = self.text_transformer(text, mask=None)

        # Fusion layer
        for fusion_layer in self.fusion_layers:
            dialogue_text_features = fusion_layer(dialogue_audio_features, dialogue_text_features)

        # concatenation layer
        x = torch.cat((dialogue_audio_features, dialogue_text_features), dim=2)

        # Output layer
        x = self.output_layer(x)

        return x
