import torch
from utils import get_text, get_utterance_with_context
from transformers import RobertaTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        super().__init__()

        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        self.mode = mode

        self.text = get_text(mode)

        # Remove corrupted multimedia files
        if self.mode == "train":
            self.text = self.text[(self.text["Dialogue_ID"] != 125) | (self.text["Utterance_ID"] != 3)]
        elif self.mode == "val":
            self.text = self.text[(self.text["Dialogue_ID"] != 110) | (self.text["Utterance_ID"] != 7)]
        elif self.mode == "test":
            self.text = self.text[(self.text["Dialogue_ID"] != 38) | (self.text["Utterance_ID"] != 4)]
            self.text = self.text[(self.text["Dialogue_ID"] != 220) | (self.text["Utterance_ID"] != 0)]

        # Map labels to class indices
        emotion_labels = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "surprise": 4, "fear": 5, "disgust": 6}
        self.text["Emotion"] = self.text["Emotion"].map(emotion_labels)

        self.dataset_size = len(self.text.index)
        print(f"Loaded {self.dataset_size} utterances for {self.mode}ing")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        df_row = self.text.iloc[idx]

        # Text
        separator = self.roberta_tokenizer.sep_token
        text = get_utterance_with_context(self.text, idx, separator)

        # Emotion
        emotion = int(df_row["Emotion"])

        return {"text": text, "emotion": emotion}

    def collate_fn(self, batch):
        # text
        text = self.roberta_tokenizer([d["text"] for d in batch], return_tensors="pt", padding="longest", truncation=True, max_length=512)
        attention_mask = text["attention_mask"].long()
        text = text["input_ids"].long()

        # emotion
        emotion = torch.tensor([d["emotion"] for d in batch], dtype=torch.int64)

        return {"text": text, "attention_mask": attention_mask, "emotion": emotion}

    def get_labels(self):
        return self.text["Emotion"].to_numpy()
