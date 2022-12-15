import torch
from ..utils.dataset_utils import get_text
import os
import torchaudio
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        if self.mode == "train":
            self.audio_path = "data/MELD.Raw/train_splits/wav"
        if self.mode == "val":
            self.audio_path = "data/MELD.Raw/dev_splits_complete/wav"
        if self.mode == "test":
            self.audio_path = "data/MELD.Raw/output_repeated_splits_test/wav"

        self.text = get_text(mode)
        self.text.drop(columns=["Speaker", "Episode", "StartTime", "EndTime"], inplace=True)
        self.sentiment_labels = self.text["Sentiment"]
        self.emotion_labels = self.text["Emotion"]

        # labels to one-hot encoding
        self.sentiment_labels_h = torch.nn.functional.one_hot(torch.tensor(pd.Categorical(self.sentiment_labels).codes, dtype=torch.int64), num_classes=3)
        self.emotion_labels_h = torch.nn.functional.one_hot(torch.tensor(pd.Categorical(self.emotion_labels).codes, dtype=torch.int64), num_classes=7)

        self.oneh2sentiment = {h:sent for h, sent in zip(self.sentiment_labels_h, self.sentiment_labels)}
        self.oneh2emotion = {h:emo for h, emo in zip(self.emotion_labels_h, self.emotion_labels)}


    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        utt_id = self.text.iloc[idx]["Utterance_ID"]
        dial_id = self.text.iloc[idx]["Dialogue_ID"]

        # Audio
        wav_path = os.path.join(self.audio_path, f"dia{dial_id}_utt{utt_id}.wav")
        audio, sr = torchaudio.load(wav_path, format="wav")
        audio = torchaudio.functional.resample(audio, sr, self.bundle.sample_rate)

        # Text
        text = self.text.iloc[idx]["Utterance"]


        return {"text":text, "audio":audio, "sample_rate":sr, "sentiment":self.sentiment_labels_h[idx], "emotion":self.emotion_labels_h[idx]}

if __name__ == "__main__":
    # Test Dataset class
    dataset = Dataset()
    data_ld = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(data_ld):
        print(data)
        break
    print(dataset[0])
