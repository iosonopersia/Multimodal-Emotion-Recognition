import torch
import torch.nn as nn


class FusionAttention(nn.Module):
    def __init__(self, embedding_size, n_head):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embedding_size, n_head, batch_first=True)
        self.linear = nn.Linear(2*embedding_size, embedding_size)
        self.relu = nn.ReLU()


    def forward(self, embedding_text, embedding_audio, mask):#, src_key_padding_mask):
        x, _ = self.multihead_attention(embedding_text, embedding_audio, embedding_text, attn_mask=mask)#, key_padding_mask=src_key_padding_mask)

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

    def forward(self, text, audio, mask):
        # Audio and text encoders
        # TODO pass the previous and next utterance to the feature extractor
        # TODO construct the mask for the audio and text encoders (starting from the lengths)

        text = text.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2)

        # (batch_size, seq_len, seq_len)
        squared_mask = mask.unsqueeze(1).repeat(1, mask.shape[1], 1)
        squared_mask = squared_mask & squared_mask.transpose(1, 2)
        squared_mask = squared_mask.repeat(8, 1, 1)
        squared_mask = squared_mask.float() # Convert from bool to float for the transformer

        text = self.text_transformer(text, mask=squared_mask)#, src_key_padding_mask=mask)
        audio = self.audio_transformer(audio, mask=squared_mask)#, src_key_padding_mask=mask)

        text = text.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2)

        # Fusion layer
        for fusion_layer in self.fusion_layers:
            text = fusion_layer(audio, text, mask=squared_mask)#, src_key_padding_mask=mask)

        # concatenation layer
        x = torch.cat((audio, text), dim=2)

        # Output layer
        x = self.output_layer(x)

        return x
