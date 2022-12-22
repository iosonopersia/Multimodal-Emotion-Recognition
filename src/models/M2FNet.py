import torch
import torch.nn as nn


class FusionAttentionLayer(nn.Module):
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

        n_layers_audio = config.AUDIO.N_LAYERS
        n_layers_text = config.TEXT.N_LAYERS

        n_encoders_audio = config.AUDIO.N_ENCODERS
        n_encoders_text = config.TEXT.N_ENCODERS

        n_fam_layers = config.FAM.N_LAYERS
        n_layers_classifier = config.CLASSIFIER.N_LAYERS
        hidden_size_classifier = config.CLASSIFIER.HIDDEN_SIZE
        output_size = config.OUTPUT_SIZE

        # Audio and text encoders
        audio_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_audio, nhead=self.n_head_audio, dropout=dropout)
        audio_encoder_norm = nn.LayerNorm(d_model_audio)
        self.audio_encoders = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer=audio_encoder_layer, norm=audio_encoder_norm, num_layers=n_layers_audio)
            for _ in range(n_encoders_audio)])

        text_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_text, nhead=self.n_head_text, dropout=dropout)
        text_encoder_norm = nn.LayerNorm(d_model_text)
        self.text_encoders = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer=text_encoder_layer, norm=text_encoder_norm, num_layers=n_layers_text)
            for _ in range(n_encoders_text)])

        # Fusion Attention layers
        self.fusion_layers = nn.ModuleList([
            FusionAttentionLayer(embedding_size=d_model_fam, n_head=self.n_head_fam, dropout=dropout)
            for _ in range(n_fam_layers)])

        # Output layers
        classifier_head = [nn.Linear(d_model_text + d_model_audio, hidden_size_classifier)]
        if n_layers_classifier > 2:
            for _ in range(n_layers_classifier - 2):
                classifier_head.append(nn.ReLU())
                classifier_head.append(nn.Linear(hidden_size_classifier, hidden_size_classifier))

        classifier_head.append(nn.ReLU())
        classifier_head.append(nn.Linear(hidden_size_classifier, output_size))

        self.output_layer = nn.Sequential(*classifier_head)

    def forward(self, text, audio, mask):
        text = text.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2)

        # Audio encoders with local skip connections
        for encoder in self.audio_encoders:
            audio = audio + encoder(audio, src_key_padding_mask=mask)

        # Text encoders with local skip connections
        for encoder in self.text_encoders:
            text = text + encoder(text, src_key_padding_mask=mask)

        text = text.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2)

        # Fusion Attention layers
        for fusion_layer in self.fusion_layers:
            text = fusion_layer(text=text, audio=audio, key_padding_mask=mask)

        # Concatenation layer
        x = torch.cat((audio, text), dim=2)

        # Fully Connected output layer
        x = self.output_layer(x)

        return x
