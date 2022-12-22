import torch
import torch.nn as nn


class FusionAttention(nn.Module):
    def __init__(self, embedding_size, n_head, dropout):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embedding_size, n_head, batch_first=True, dropout=dropout)
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
        dropout = config.DROPOUT
        d_model_audio = config.AUDIO.EMBEDDING_SIZE
        d_model_text = config.TEXT.EMBEDDING_SIZE
        d_model_fam = config.FAM.EMBEDDING_SIZE
        self.n_head_audio = config.AUDIO.N_HEAD
        self.n_head_text = config.TEXT.N_HEAD
        self.n_head_fam = config.FAM.N_HEAD
        self.n_layers_audio = config.AUDIO.N_LAYERS
        self.n_layers_text = config.TEXT.N_LAYERS
        n_fam_layers = config.FAM.N_LAYERS
        n_layers_classifier = config.CLASSIFIER.N_LAYERS
        hidden_size_classifier = config.CLASSIFIER.HIDDEN_SIZE
        output_size = config.OUTPUT_SIZE

        #Audio and text encoders [Dialogue-level]
        self.audio_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model_audio, nhead=self.n_head_audio, dropout=dropout)
        ] * self.n_layers_audio)
        self.audio_encoder_norm = nn.LayerNorm(d_model_audio)

        self.text_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model_text, nhead=self.n_head_text, dropout=dropout)
        ] * self.n_layers_text)
        self.text_encoder_norm = nn.LayerNorm(d_model_text)

        # Fusion layer [Dialogue-level]
        self.fusion_layers = nn.ModuleList([
            FusionAttention(embedding_size=d_model_fam, n_head=self.n_head_fam, dropout=dropout)
        ] * n_fam_layers)

        # Output layer [Dialogue-level]
        classifier_head = [nn.Linear(d_model_text + d_model_audio, hidden_size_classifier)]
        if n_layers_classifier > 2:
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
        squared_mask_text = squared_mask.repeat(self.n_head_text, 1, 1).float()
        squared_mask_audio = squared_mask.repeat(self.n_head_audio, 1, 1).float()
        squared_mask_fam = squared_mask.repeat(self.n_head_fam, 1, 1).float()

        # Add skip connections to audio encoders
        for layer in self.audio_encoder_layers:
            audio = audio + layer(audio, src_mask=squared_mask_audio)#, src_key_padding_mask=mask)
        audio = self.audio_encoder_norm(audio)

        # Add skip connections to text encoders
        for layer in self.text_encoder_layers:
            text = text + layer(text, src_mask=squared_mask_text)#, src_key_padding_mask=mask)
        text = self.text_encoder_norm(text)

        text = text.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2)

        # Fusion layer
        for fusion_layer in self.fusion_layers:
            text = fusion_layer(audio, text, mask=squared_mask_fam)#, src_key_padding_mask=mask)

        # concatenation layer
        x = torch.cat((audio, text), dim=2)

        # Output layer
        x = self.output_layer(x)

        return x
