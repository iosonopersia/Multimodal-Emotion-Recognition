import torch
from models.FeatureExtractor import FeatureExtractor
from utils import get_config, apply_padding
from utils.dataset_utils import get_text
import os
import torchaudio
from transformers import RobertaTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        super().__init__()

        config = get_config()
        self.ffmpeg_sr = config.AUDIO.ffmpeg_sr
        self.wav2vec_sr = config.AUDIO.wav2vec_sr

        self.roberta_encoder = RobertaTokenizer.from_pretrained('roberta-base')

        self.feature_embedding_model = FeatureExtractor(torch.device("cuda:0"))

        self.mode = mode
        if self.mode == "train":
            self.audio_path = "data/MELD.Raw/train_splits/wav"
        if self.mode == "val":
            self.audio_path = "data/MELD.Raw/dev_splits_complete/wav"
        if self.mode == "test":
            self.audio_path = "data/MELD.Raw/output_repeated_splits_test/wav"

        self.text = get_text(mode)
        self.sentiment_labels = self.text["Sentiment"]
        self.emotion_labels = self.text["Emotion"]

        # labels to one-hot encoding
        sentiment_labels = ["neutral", "negative", "positive"]
        num_sentiments = len(sentiment_labels)
        sent_oneh = torch.nn.functional.one_hot(torch.arange(0, num_sentiments, dtype=torch.int64), num_classes=num_sentiments)
        self.sentiment2oneh = {sent: sent_oneh for sent, sent_oneh in zip(sentiment_labels, sent_oneh)}

        emotion_labels = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
        num_emotions = len(emotion_labels)
        emo_oneh = torch.nn.functional.one_hot(torch.arange(0, num_emotions, dtype=torch.int64), num_classes=num_emotions)
        self.emotion2oneh = {emo: emo_oneh for emo, emo_oneh in zip(emotion_labels, emo_oneh)}

        # Inversed dictionaries
        self.oneh2sentiment = {h: sent for h, sent in self.sentiment2oneh.items()}
        self.oneh2emotion = {h: emo for h, emo in self.emotion2oneh.items()}

        self.dialogue_ids = self.text["Dialogue_ID"].unique()
        print(f"Loaded {len(self.dialogue_ids)} dialogues for {self.mode}ing")

    def __len__(self):
        return len(self.dialogue_ids)

    def __getitem__(self, idx):
        dialogue_id = self.dialogue_ids[idx]

        utterances = self.text[self.text["Dialogue_ID"] == dialogue_id].sort_values(by="Utterance_ID")

        text = []
        audio = []
        sentiment = []
        emotion = []
        for _, utterance in utterances.iterrows():
            utterance_id = utterance["Utterance_ID"]

            if self.mode == "train":
                if (dialogue_id, utterance_id) in {(125, 3)}:
                    # This utterance video/audio is corrupted :-(
                    continue
            elif self.mode == "val":
                if (dialogue_id, utterance_id) in {(110, 7)}:
                # This utterance video/audio is corrupted :-(
                continue

            # Audio
            _wav_path = os.path.join(os.path.abspath(self.audio_path), f"dia{dialogue_id}_utt{utterance_id}.wav")
            _audio, _ = torchaudio.load(_wav_path, format="wav")

            # Text
            _text = utterance["Utterance"]

            # Sentiment
            _sentiment = utterance["Sentiment"]

            # Emotion
            _emotion = utterance["Emotion"]

            audio.append(_audio)
            text.append(_text)
            sentiment.append(self.sentiment2oneh[_sentiment])
            emotion.append(self.emotion2oneh[_emotion])

        # Tokenize text
        text = self.roberta_encoder(text, return_tensors="pt", padding="longest")

        # Resample to 16kHz for wav2vec2
        if self.wav2vec_sr != self.ffmpeg_sr:
            audio = [torchaudio.functional.resample(a, self.ffmpeg_sr, self.wav2vec_sr) for a in audio]

        return {"text": text, "audio": audio, "sentiment": sentiment, "emotion": emotion}

    def my_collate_fn(self, batch):
        # text
        text = [d["text"] for d in batch]

        # audio
        audio = [d["audio"] for d in batch]

        # sentiment
        sentiment = [torch.stack(d["sentiment"], dim=0).unsqueeze(dim=0) for d in batch]
        sentiment, _ = apply_padding(sentiment)

        # emotion
        emotion = [torch.stack(d["emotion"], dim=0).unsqueeze(dim=0) for d in batch]
        emotion, _ = apply_padding(emotion)

        text, audio = self.feature_embedding_model(text, audio)

        return {"text": text, "audio": audio, "sentiment": sentiment, "emotion": emotion}
