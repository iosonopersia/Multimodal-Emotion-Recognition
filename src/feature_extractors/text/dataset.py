import torch
from utils import get_text, get_utterance_with_context
from transformers import RobertaTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        super().__init__()

        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        self.mode = mode

        self.text = get_text(mode)

        # Map labels to class indices
        emotion_labels = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "surprise": 4, "fear": 5, "disgust": 6}
        self.text["Emotion"] = self.text["Emotion"].map(emotion_labels)

        # Count how many dialogues there are
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
