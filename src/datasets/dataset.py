import torch
from utils.dataset_utils import get_text
import os
import torchaudio
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
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

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        utterances = self.text[self.text["Dialogue_ID"] == idx].sort_values(by="Utterance_ID")

        dialogue_id = idx
        text = []
        audio = []
        sr = []
        sentiment = []
        emotion = []
        for _, utterance in utterances.iterrows():
            utterance_id = utterance["Utterance_ID"]

            # Audio
            _wav_path = os.path.join(os.path.abspath(self.audio_path), f"dia{dialogue_id}_utt{utterance_id}.wav")
            _audio, _sr = torchaudio.load(_wav_path, format="wav")

            # Text
            _text = utterance["Utterance"]

            # Sentiment
            _sentiment = utterance["Sentiment"]

            # Emotion
            _emotion = utterance["Emotion"]

            audio.append(_audio)
            sr.append(_sr)
            text.append(_text)
            sentiment.append(self.sentiment2oneh[_sentiment])
            emotion.append(self.emotion2oneh[_emotion])

        return {"text": text, "audio": audio, "sample_rate": sr, "sentiment": sentiment, "emotion": emotion}
