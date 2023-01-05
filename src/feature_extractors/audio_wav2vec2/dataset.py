import torch
from utils import get_text
import os
import torchaudio
from torchaudio.pipelines import WAV2VEC2_BASE as WAV2VEC2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        super().__init__()
        self.mode = mode
        if self.mode == "train":
            self.audio_path = "data/MELD.Raw/train_splits/wav"
        elif self.mode == "val":
            self.audio_path = "data/MELD.Raw/dev_splits_complete/wav"
        elif self.mode == "test":
            self.audio_path = "data/MELD.Raw/output_repeated_splits_test/wav"
        self.audio_path = os.path.abspath(self.audio_path)

        self.text = get_text(mode)

        # Map labels to class indices
        emotion_labels = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "surprise": 4, "fear": 5, "disgust": 6}
        self.text["Emotion"] = self.text["Emotion"].map(emotion_labels)

        self.dataset_size = len(self.text.index)
        print(f"Loaded {self.dataset_size} utterances for {self.mode}ing")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        df_row = self.text.iloc[idx]

        # Audio
        dialogue_id = df_row["Dialogue_ID"]
        utterance_id = df_row["Utterance_ID"]
        wav_path = os.path.join(self.audio_path, f"dia{dialogue_id}_utt{utterance_id}.wav")
        audio, sr = torchaudio.load(wav_path, format="wav")

        # Resample to 16kHz for wav2vec2
        if WAV2VEC2.sample_rate != sr:
            audio = torchaudio.functional.resample(audio, sr, WAV2VEC2.sample_rate)
        audio = audio.squeeze(dim=0)

        # Truncate to 10 seconds
        if audio.shape[0] > 10 * WAV2VEC2.sample_rate:
            audio = audio[:10 * WAV2VEC2.sample_rate]

        # Emotion
        emotion = int(df_row["Emotion"])

        return {"idx": idx, "audio": audio, "emotion": emotion}

    def get_labels(self):
        return self.text["Emotion"].to_numpy()


def collate_fn(batch):
    # idx
    idx = [dialogue["idx"] for dialogue in batch]

    # Pad audio to the longest audio in the batch
    audio_lengths = torch.tensor([dialogue["audio"].shape[0] for dialogue in batch], dtype=torch.int64)
    max_length = audio_lengths.max().item()
    audio = [torch.nn.functional.pad(dialogue["audio"], (0, max_length - dialogue["audio"].shape[0]), mode="constant", value=0.0) for dialogue in batch]
    audio = torch.stack(audio)

    # emotion
    emotion = torch.tensor([dialogue["emotion"] for dialogue in batch], dtype=torch.int64)

    return {"idx": idx, "audio": audio, "lengths": audio_lengths, "emotion": emotion}
