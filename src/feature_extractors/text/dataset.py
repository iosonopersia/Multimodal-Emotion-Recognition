import torch
from utils import get_text
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
        main_utterance = self.text.iloc[idx]
        dialogue_id = int(main_utterance["Dialogue_ID"])
        main_utterance_id = int(main_utterance["Utterance_ID"])

        # Get the previous and next utterance IDs
        dialogue = self.text[self.text["Dialogue_ID"] == dialogue_id]
        dia_utt_ids = sorted(dialogue["Utterance_ID"].to_list())
        try:
            main_utt_pos_in_dialogue = dia_utt_ids.index(main_utterance_id)
        except ValueError:
            raise ValueError(f"Utterance ID {main_utterance_id} not found in dialogue ID {dialogue_id}")
        prev_utterance_id = dia_utt_ids[main_utt_pos_in_dialogue - 1] if main_utt_pos_in_dialogue > 0 else None
        next_utterance_id = dia_utt_ids[main_utt_pos_in_dialogue + 1] if main_utt_pos_in_dialogue < len(dia_utt_ids) - 1 else None

        # Concatenate the previous and next utterances
        utterances = [' ', main_utterance["Utterance"], ' ']
        if prev_utterance_id is not None:
            prev_utterance_row = dialogue[dialogue["Utterance_ID"] == prev_utterance_id].iloc[0]
            utterances[0] = prev_utterance_row["Utterance"]

        if next_utterance_id is not None:
            next_utterance_row = dialogue[dialogue["Utterance_ID"] == next_utterance_id].iloc[0]
            utterances[2] = next_utterance_row["Utterance"]

        # Text
        separator = f" {self.roberta_tokenizer.sep_token} "
        text = separator.join(utterances)

        # Emotion
        emotion = int(main_utterance["Emotion"])

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
