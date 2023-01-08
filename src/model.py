import torch
import torch.nn as nn


class FusionAttentionModule(nn.Module):
    def __init__(self, embedding_size, n_head, dropout):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embedding_size, n_head, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(2*embedding_size, embedding_size)
        self.relu = nn.ReLU()


    def forward(self, text, audio, key_padding_mask):
        x, _ = self.multihead_attention(query=text, key=audio, value=text, key_padding_mask=key_padding_mask)

        x = torch.cat((x, text), dim=2)
        x = self.relu(x)
        x = self.linear(x)
        x = self.relu(x)
        return x


class M2FNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.audio_enabled = config.AUDIO.enabled
        self.text_enabled = config.TEXT.enabled
        self.fam_enabled = config.FAM.enabled

        if not self.audio_enabled and not self.text_enabled:
            raise ValueError("At least one of audio and text must be enabled!")
        if self.fam_enabled and not (self.audio_enabled and self.text_enabled):
            raise ValueError("Fusion Attention Module can only be used with both audio and text enabled!")

        d_model_audio = config.AUDIO.embedding_size
        d_model_text = config.TEXT.embedding_size
        d_model_fam = config.FAM.embedding_size

        self.n_head_audio = config.AUDIO.n_head
        self.n_head_text = config.TEXT.n_head
        self.n_head_fam = config.FAM.n_head

        n_layers_audio = config.AUDIO.n_encoder_layers
        n_layers_text = config.TEXT.n_encoder_layers
        n_layers_fam = config.FAM.n_layers
        n_layers_classifier = config.CLASSIFIER.n_layers

        n_transformers_audio = config.AUDIO.n_transformers
        n_transformers_text = config.TEXT.n_transformers

        hidden_size_classifier = config.CLASSIFIER.hidden_size
        output_size_classifier = config.CLASSIFIER.output_size

        dropout = config.dropout
        self.dropout = nn.Dropout(dropout)

        if self.audio_enabled:
            # Audio encoders
            audio_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_audio, nhead=self.n_head_audio, dropout=dropout)
            audio_encoder_norm = nn.LayerNorm(d_model_audio)
            self.audio_encoders = nn.ModuleList([
                nn.TransformerEncoder(encoder_layer=audio_encoder_layer, norm=audio_encoder_norm, num_layers=n_layers_audio)
                for _ in range(n_transformers_audio)])

            # Audio projection layer
            self.audio_proj = nn.Linear(d_model_audio, d_model_fam)


        if self.text_enabled:
            # Text encoders
            text_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_text, nhead=self.n_head_text, dropout=dropout)
            text_encoder_norm = nn.LayerNorm(d_model_text)
            self.text_encoders = nn.ModuleList([
                nn.TransformerEncoder(encoder_layer=text_encoder_layer, norm=text_encoder_norm, num_layers=n_layers_text)
                for _ in range(n_transformers_text)])

            # Text projection layer
            self.text_proj = nn.Linear(d_model_text, d_model_fam)

        if self.fam_enabled:
            # Fusion Attention Module layers
            self.fusion_layers = nn.ModuleList([
                FusionAttentionModule(embedding_size=d_model_fam, n_head=self.n_head_fam, dropout=dropout)
                for _ in range(n_layers_fam)])

        # Output layers
        cls_input_size = 2*d_model_fam if (self.audio_enabled and self.text_enabled) else d_model_fam
        classifier_head = [nn.Linear(cls_input_size, hidden_size_classifier)]
        if n_layers_classifier > 2:
            for _ in range(n_layers_classifier - 2):
                classifier_head.append(nn.ReLU())
                classifier_head.append(nn.Linear(hidden_size_classifier, hidden_size_classifier))

        classifier_head.append(nn.ReLU())
        classifier_head.append(self.dropout)
        classifier_head.append(nn.Linear(hidden_size_classifier, output_size_classifier))

        self.output_layer = nn.Sequential(*classifier_head)

    def forward(self, text, audio, mask):
        if self.audio_enabled:
            # Audio encoders with local skip connections
            audio = audio.permute(1, 0, 2)
            for encoder in self.audio_encoders:
                audio = audio + encoder(audio, src_key_padding_mask=mask)
            audio = audio.permute(1, 0, 2)

            # Audio projection layer
            audio = self.dropout(audio)
            audio = self.audio_proj(audio)
            audio = self.dropout(audio)

        if self.text_enabled:
            # Text encoders with local skip connections
            text = text.permute(1, 0, 2)
            for encoder in self.text_encoders:
                text = text + encoder(text, src_key_padding_mask=mask)
            text = text.permute(1, 0, 2)

            # Text projection layer
            text = self.dropout(text)
            text = self.text_proj(text)
            text = self.dropout(text)

        if self.fam_enabled:
            # Fusion Attention layers
            for fusion_layer in self.fusion_layers:
                text = fusion_layer(text=text, audio=audio, key_padding_mask=mask)
                text = self.dropout(text)

            # Concatenation layer
            x = torch.cat((audio, text), dim=2)
        else:
            # Concatenation layer
            if self.audio_enabled and self.text_enabled:
                x = torch.cat((audio, text), dim=2)
            else:
                x = text if self.text_enabled else audio

        # Fully Connected output layer
        x = self.output_layer(x)

        return x
