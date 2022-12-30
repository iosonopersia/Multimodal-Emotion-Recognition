import torch
from utils import get_config, apply_padding
from utils.dataset_utils import get_text, get_utterance_with_context
import os
import torchaudio
from transformers import RobertaTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        super().__init__()

        config = get_config()
        self.ffmpeg_sr = config.AUDIO.ffmpeg_sr
        self.wav2vec_sr = config.AUDIO.wav2vec_sr

        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        self.mode = mode
        if self.mode == "train":
            self.audio_path = "data/MELD.Raw/train_splits/wav"
        if self.mode == "val":
            self.audio_path = "data/MELD.Raw/dev_splits_complete/wav"
        if self.mode == "test":
            self.audio_path = "data/MELD.Raw/output_repeated_splits_test/wav"

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

            if self.mode == "train":
                if (dialogue_id, utterance_id) in {(125, 3)}:
                    # This utterance video/audio is corrupted :-(
                    continue
            elif self.mode == "val":
                if (dialogue_id, utterance_id) in {(110, 7)}:
                    # This utterance video/audio is corrupted :-(
                    continue
            elif self.mode == "test":
                if (dialogue_id, utterance_id) in {(38,4),(220,0)}:
                    # This utterance video/audio is corrupted :-(
                    continue

            # Audio
            _wav_path = os.path.join(os.path.abspath(self.audio_path), f"dia{dialogue_id}_utt{utterance_id}.wav")
            _audio, _ = torchaudio.load(_wav_path, format="wav")

            # Text
            current_row_mask = (self.text["Dialogue_ID"] == dialogue_id) & (self.text["Utterance_ID"] == utterance_id)
            assert len(self.text[current_row_mask].index) == 1
            df_row_idx = self.text[current_row_mask].index[0]

            separator = self.roberta_tokenizer.sep_token
            _text = get_utterance_with_context(self.text, df_row_idx, separator)

            # Emotion
            _emotion = utterance["Emotion"]
            _emotion = torch.tensor([_emotion])

            audio.append(_audio)
            text.append(_text)
            emotion.append(_emotion)

        # Tokenize text
        text = self.roberta_tokenizer(text, return_tensors="pt", padding="longest")

        # Resample to 16kHz for wav2vec2
        if self.wav2vec_sr != self.ffmpeg_sr:
            audio = [torchaudio.functional.resample(a, self.ffmpeg_sr, self.wav2vec_sr) for a in audio]

        return {"text": text, "audio": audio, "emotion": emotion}

    def my_collate_fn(self, batch):
        # text
        text = [d["text"] for d in batch]

        # audio
        audio = [d["audio"] for d in batch]

        # emotion
        emotion = [torch.stack(d["emotion"], dim=0).unsqueeze(dim=0) for d in batch]
        emotion, _ = apply_padding(emotion, padding_value=-1) # -1 class index is ignored during loss computation
        emotion = emotion.squeeze(dim=2)

        return {"text": text, "audio": audio, "emotion": emotion}

    def get_labels(self):
        return self.text["Emotion"].to_numpy()
