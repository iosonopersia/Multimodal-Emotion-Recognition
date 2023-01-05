import torch
from utils import get_config, apply_padding, get_text
import os
import pickle


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        super().__init__()
        config = get_config()

        self.mode = mode

        text_embeddings_path = os.path.join(os.path.abspath(config.embeddings.text), f"{mode}.pkl")
        audio_embeddings_path = os.path.join(os.path.abspath(config.embeddings.audio), f"{mode}.pkl")
        self.text_embeddings = pickle.load(open(text_embeddings_path, "rb"))
        self.audio_embeddings = pickle.load(open(audio_embeddings_path, "rb"))

        self.text = get_text(mode)

        # Map labels to class indices
        emotion_labels = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "surprise": 4, "fear": 5, "disgust": 6}
        self.text["Emotion"] = self.text["Emotion"].map(emotion_labels)

        # Count how many dialogues there are
        self.dialogue_ids = self.text["Dialogue_ID"].unique()
        print(f"Loaded {len(self.dialogue_ids)} dialogues for {self.mode}ing")

    def __len__(self):
        return len(self.dialogue_ids)

    def __getitem__(self, idx):
        dialogue_id = self.dialogue_ids[idx]

        utterances = self.text[self.text["Dialogue_ID"] == dialogue_id].sort_values(by="Utterance_ID")

        text = []
        audio = []
        emotion = []
        for _, utterance in utterances.iterrows():
            utterance_id = utterance["Utterance_ID"]

            current_row_mask = (self.text["Dialogue_ID"] == dialogue_id) & (self.text["Utterance_ID"] == utterance_id)
            assert len(self.text[current_row_mask].index) == 1
            df_row_idx = self.text[current_row_mask].index[0]

            # Audio
            _audio = self.audio_embeddings[df_row_idx]

            # Text
            _text = self.text_embeddings[df_row_idx]

            # Emotion
            _emotion = utterance["Emotion"]
            _emotion = torch.tensor([_emotion])

            audio.append(_audio)
            text.append(_text)
            emotion.append(_emotion)


        text = torch.stack(text)
        audio = torch.stack(audio)

        return {"text": text, "audio": audio, "emotion": emotion}

    def get_labels(self):
        return self.text["Emotion"].to_numpy()


def collate_fn(batch):
    # text
    text = [dialogue["text"].unsqueeze(dim=0) for dialogue in batch]
    text, _ = apply_padding(text)

    # audio
    audio = [dialogue["audio"].unsqueeze(dim=0) for dialogue in batch]
    audio, _ = apply_padding(audio)

    # emotion
    emotion = [torch.stack(dialogue["emotion"], dim=0).unsqueeze(dim=0) for dialogue in batch]
    emotion, _ = apply_padding(emotion, padding_value=-1) # -1 class index is ignored during loss computation
    emotion = emotion.squeeze(dim=2)

    # padding mask
    padding_mask = torch.zeros_like(emotion, dtype=torch.bool)
    padding_mask[emotion == -1] = True

    return {"text": text, "audio": audio, "padding_mask": padding_mask, "emotion": emotion}
