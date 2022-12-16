import torch
import torchaudio
import torch.nn as nn


class feature_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec = self.bundle.get_model()


    def forward(self, text, audio, sr):
        with torch.inference_mode():
            # text embeddings
            tokens = self.roberta.encode(text)
            text_features = self.roberta.extract_features(tokens)

            # audio embeddings
            audio = torchaudio.functional.resample(audio, sr, self.bundle.sample_rate)
            audio_features, _ = self.wav2vec.extract_features(audio)
        return {"text_features": text_features, "audio_features":audio_features}

class FAM (nn.Module):
    def __init__(self, embedding_size, n_head):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embedding_size, n_head)
        self.linear = nn.Linear(2*embedding_size, embedding_size)
        self.relu = nn.ReLU()


    def forward(self, embedding_text, embedding_audio):
        x = self.multihead_attention(embedding_text, embedding_audio, embedding_text)
        x = torch.cat ((x, embedding_text), dim=2)
        x = self.linear(self.relu(x))
        return x

class MERmodel(nn.Module):
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
        output_size = config.OUTPUT_SIZE

        #Feature extractor [Utterance-level]
        self.feature_extractor = feature_extractor()

        #Audio and text encoders [Dialogue-level]
        audio_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_audio, nhead=n_head_audio)
        self.audio_transformer = nn.TransformerEncoder(audio_encoder_layer, num_layers=n_layers_audio)

        text_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_text, nhead=n_head_text)
        self.text_transformer = nn.TransformerEncoder(text_encoder_layer, num_layers=n_layers_text)

        # Fusion layer [Dialogue-level]
        self.fusion_layer = FAM(embedding_size=d_model_audio, n_head=n_head_audio)

        # Output layer [Dialogue-level]
        self.output_layer = nn.Sequential (nn.Linear(d_model_audio, 32), nn.ReLU(), nn.Linear(32, output_size))




    def forward(self, dialogue_text, dialogue_audio, dialogue_audio_sr):
        # Feature extraction
        # TODO concat or merge output of the extractors
        # TODO pass the previous and next utterance to the feature extractor
        utterance_text_features = []
        utterance_audio_features = []
        for i in range(len(dialogue_text)):
            features = self.feature_extractor(dialogue_text[i], dialogue_audio[i], dialogue_audio_sr[i])
            utterance_text_features.append(features["text_features"])
            utterance_audio_features.append(features["audio_features"])

        #concatenate the utterance features(?)
        # TODO problem with the length of the concatenation
        # dialogue_text_features = torch.cat(utterance_text_features, dim=0)
        # dialogue_audio_features = torch.cat(utterance_audio_features, dim=0)

        # Audio and text encoders
        dialogue_audio_features = self.audio_transformer(dialogue_audio_features)
        dialogue_text_features = self.text_transformer(dialogue_text_features)

        # Fusion layer
        dialogue_features = self.fusion_layer(dialogue_audio_features, dialogue_text_features)

        # concatenation layer
        x = torch.cat((dialogue_features, dialogue_audio_features))

        # Output layer
        x = self.output_layer(x)

        return x









