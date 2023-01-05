import torch
from utils import get_text, get_utterance_with_context
from transformers import RobertaTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        super().__init__()
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.sep_token = roberta_tokenizer.sep_token

        self.mode = mode

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

        # Text
        text = get_utterance_with_context(self.text, idx, self.sep_token)

        # Emotion
        emotion = int(df_row["Emotion"])

        return {"idx": idx, "text": text, "emotion": emotion}

    def get_labels(self):
        return self.text["Emotion"].to_numpy()


def collate_fn(batch):
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # idx
    idx = [dialogue["idx"] for dialogue in batch]

    # text
    text = roberta_tokenizer([dialogue["text"] for dialogue in batch], return_tensors="pt", padding="longest", truncation=True, max_length=512)
    attention_mask = text["attention_mask"].long()
    text = text["input_ids"].long()

    # emotion
    emotion = torch.tensor([dialogue["emotion"] for dialogue in batch], dtype=torch.int64)

    return {"idx": idx, "text": text, "attention_mask": attention_mask, "emotion": emotion}
