import random
import torch
from utils import get_text
import os
import torchaudio
import matplotlib.pyplot as plt
import warnings
from PIL import Image
import numpy as np
import librosa
from tqdm import tqdm
import pandas as pd


class DatasetMelAudio(torch.utils.data.Dataset):
    def __init__(self, mode="train", config=None):
        super().__init__()

        self.MAX_AUDIO_LENGTH = 10 #seconds
        self.config = config
        self.len_triplet_picking = 100
        self.mode = mode
        if self.mode == "train":
            self.audio_path = "data/MELD.Raw/train_splits/wav"
            self.mel_spectogram_cache = "data/MELD.Raw/train_splits/mel_spectograms"

        if self.mode == "val":
            self.audio_path = "data/MELD.Raw/dev_splits_complete/wav"
            self.mel_spectogram_cache = "data/MELD.Raw/dev_splits_complete/mel_spectograms"
        if self.mode == "test":
            self.audio_path = "data/MELD.Raw/output_repeated_splits_test/wav"
            self.mel_spectogram_cache = "data/MELD.Raw/output_repeated_splits_test/mel_spectograms"


        os.makedirs(self.mel_spectogram_cache, exist_ok=True)
        self.text = get_text(mode)
        if self.config.DEBUG.enabled == True:
            self.text = self.text.iloc[0:self.config.DEBUG.num_samples]

        # Map labels to class indices
        self.emotion_labels = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "surprise": 4, "fear": 5, "disgust": 6}
        self.text["Emotion"] = self.text["Emotion"].map(self.emotion_labels)

        # Count how many dialogues there are
        self.dialogue_ids = self.text["Dialogue_ID"].unique()
        print(f"Loaded {len(self.dialogue_ids)} dialogues for {self.mode}ing")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):

        utterance = self.text.iloc[idx]
        dialogue_id = utterance["Dialogue_ID"]
        utterance_id = utterance["Utterance_ID"]

        # Audio
        wav_path = os.path.join(os.path.abspath(self.audio_path), f"dia{dialogue_id}_utt{utterance_id}.wav")

        # Emotion
        emotion = utterance["Emotion"]
        emotion = torch.tensor([emotion])

        audio_mel_spectogram = self.get_mel_spectogram(wav_path)

        return {"audio_mel_spectogram": audio_mel_spectogram, "emotion": emotion}

    def get_labels(self):
        return self.text["Emotion"].to_numpy()

    # from https://github.com/liuxubo717/cl4ac
    def _get_mel_spectogram(self, audio_data: np.ndarray, sr: int, nb_fft: int,
                       hop_size: int, nb_mels: int, f_min: float,
                       f_max: float, htk: bool, power: float, norm: bool,
                       window_function: str, center: bool) -> np.ndarray:

        y = audio_data/abs(audio_data).max()
        mel_bands = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=nb_fft, hop_length=hop_size, win_length=nb_fft,
            window=window_function, center=center, power=power, n_mels=nb_mels,
            fmin=f_min, fmax=f_max, htk=htk, norm=norm).T

        return np.log(mel_bands + np.finfo(float).eps)

    def get_mel_spectogram(self, audio_path):
        '''
        transform audio into 2D Mel Spectrogram (RGB) :
            the Short Time Fourier transform (STFT) is used with the frame length of 400 samples (25 ms) and hop length of
            160 samples (10ms).
            We also use 128 Mel filter banks to generate the Mel Spectrogram

        '''
        save_to_cache = True
        augmentation = False

        # take last part of the audio_path
        audio_path_cache = os.path.split(audio_path)[-1]
        # remove extension
        audio_path_cache = audio_path_cache.split(".")[0]
        audio_path_cache = os.path.join(self.mel_spectogram_cache, f"{audio_path_cache}.png")

        if os.path.exists(audio_path_cache):
            # load from cache
            # Open the png image using PIL
            audio_mel_spectogram = Image.open(audio_path_cache)
            # Convert the PIL image to a NumPy array
            audio_mel_spectogram = np.array(audio_mel_spectogram)
            # Convert the NumPy array to a tensor
            audio_mel_spectogram = torch.from_numpy(audio_mel_spectogram).to(torch.float32) / 255
            audio_mel_spectogram = audio_mel_spectogram.repeat(3, 1, 1)
            # plt.imshow( audio_mel_spectogram.permute(1,2,0).detach().cpu().numpy())
            return audio_mel_spectogram

        audio, sr = torchaudio.load(audio_path, format="wav", normalize=True)
        if augmentation and self.mode=="train":
            noise = torch.randn(size=(audio.shape[0], audio.shape[1])) / 300
            audio = audio + noise
        audio = torch.nn.functional.pad(audio, (0, self.MAX_AUDIO_LENGTH * sr - audio.shape[1]), mode='constant', value=0)
        audio_mel_spectogram = self._get_mel_spectogram(audio.numpy(), sr, hop_size=160, nb_fft=400, nb_mels=128, norm=1, power=1, center=True, htk=False, f_min=0, f_max=None, window_function="hann")
        audio_mel_spectogram = torch.tensor(audio_mel_spectogram).permute(2,0,1)
        audio_mel_spectogram = (audio_mel_spectogram - audio_mel_spectogram.min()) / (audio_mel_spectogram.max() - audio_mel_spectogram.min())

        # save to cache
        if save_to_cache:
            audio_mel_spectogram_save = audio_mel_spectogram.permute(1, 2, 0) * 255
            # plt.imshow( audio_mel_spectogram_save )
            audio_mel_spectogram_save = audio_mel_spectogram_save.numpy()
            # plt.imshow( audio_mel_spectogram_save )
            audio_mel_spectogram_save = audio_mel_spectogram_save.astype(np.uint8)
            # plt.imshow( audio_mel_spectogram_save )

            audio_mel_spectogram_save = Image.fromarray(audio_mel_spectogram_save.squeeze(), mode="L")
            audio_mel_spectogram_save.save(audio_path_cache, mode="L")

        audio_mel_spectogram = audio_mel_spectogram.repeat(3, 1, 1)

        # plot
        # plt.imshow( audio_mel_spectogram.permute(1, 2, 0) )

        return audio_mel_spectogram

    def compute_distance(self, anchor, sample):
        distance = torch.norm(anchor - sample)
        return distance

    @torch.no_grad()
    def get_batched_triplets(self, batch_size, model, mining_type="random", margin=1):
        '''
        mining_type = "hard", "semi-hard", "random"
        '''
        if mining_type not in ["hard", "semi-hard", "random"]:
            raise ValueError("mining_type must be 'hard', 'semi-hard' or 'random'")

        device = next(model.parameters()).device
        model.eval()

        if mining_type == "random":
            anchors, positives, negatives = self.mine_random_triplets(batch_size)
        elif mining_type == "semi-hard":
            anchors, positives, negatives = self.mine_semihard_triplets(batch_size, model, device, margin)
        elif mining_type == "hard":
            anchors, positives, negatives = self.mine_hard_triplets(batch_size, model, device)

        return {"anchor": anchors, "positive": positives, "negative": negatives}

    @torch.no_grad()
    def mine_random_triplets(self, batch_size):
        anchors = []
        positives = []
        negatives = []
        # take batch_size valid triples
        for i in range(batch_size):
            #anchors
            # for imbalance dataset
            emotion = random.choice(list(self.emotion_labels.values()))
            anchor_utterance = self.text[self.text["Emotion"]==emotion].sample()

            anchor_dialogue_id = anchor_utterance["Dialogue_ID"].iloc[0]
            anchor_utterance_id = anchor_utterance["Utterance_ID"].iloc[0]
            anchor = self.get_mel_spectogram(os.path.join(os.path.abspath(self.audio_path), f"dia{anchor_dialogue_id}_utt{anchor_utterance_id}.wav"))
            anchors.append(anchor)

            # positives
            positive_utterance = self.text[self.text["Emotion"]==emotion].sample()
            while ((positive_utterance["Emotion"].iloc[0] != anchor_utterance["Emotion"].iloc[0]) or (positive_utterance["Dialogue_ID"].iloc[0]  == anchor_utterance["Dialogue_ID"].iloc[0]  and positive_utterance["Utterance_ID"].iloc[0]  == anchor_utterance["Utterance_ID"].iloc[0] )):
                positive_utterance = self.text[self.text["Emotion"]==emotion].sample()
            positive_dialogue_id = positive_utterance["Dialogue_ID"].iloc[0]
            positive_utterance_id = positive_utterance["Utterance_ID"].iloc[0]
            positive = self.get_mel_spectogram(os.path.join(os.path.abspath(self.audio_path), f"dia{positive_dialogue_id}_utt{positive_utterance_id}.wav"))
            positives.append(positive)

            # negatives
            negative_utterance = self.text[self.text["Emotion"]!=emotion].sample()
            while (negative_utterance["Emotion"].iloc[0] == anchor_utterance["Emotion"].iloc[0]):
                negative_utterance = self.text[self.text["Emotion"]!=emotion].sample()
            negative_dialogue_id = negative_utterance["Dialogue_ID"].iloc[0]
            negative_utterance_id = negative_utterance["Utterance_ID"].iloc[0]
            negative = self.get_mel_spectogram(os.path.join(os.path.abspath(self.audio_path), f"dia{negative_dialogue_id}_utt{negative_utterance_id}.wav"))
            negatives.append(negative)

        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)
        return anchors, positives, negatives

    @torch.no_grad()
    def mine_semihard_triplets(self, batch_size, model, device, margin):
        anchors = []
        positives = []
        negatives = []
        # take batch_size valid triples
        for i in range(batch_size):
            hard_negative = False
            count = 0 # count the number of tries
            while hard_negative == False:
                count += 1
                #anchors
                # for imbalance dataset
                emotion = random.choice(list(self.emotion_labels.values()))
                anchor_utterance = self.text[self.text["Emotion"]==emotion].sample()
                anchor_dialogue_id = anchor_utterance["Dialogue_ID"].iloc[0]
                anchor_utterance_id = anchor_utterance["Utterance_ID"].iloc[0]
                anchor = self.get_mel_spectogram(os.path.join(os.path.abspath(self.audio_path), f"dia{anchor_dialogue_id}_utt{anchor_utterance_id}.wav"))

                # positives
                positive_utterance = self.text[self.text["Emotion"]==emotion].sample()
                while (positive_utterance["Emotion"].iloc[0] != anchor_utterance["Emotion"].iloc[0]) or (positive_utterance["Dialogue_ID"].iloc[0] == anchor_utterance["Dialogue_ID"].iloc[0]  and positive_utterance["Utterance_ID"].iloc[0]  == anchor_utterance["Utterance_ID"].iloc[0] ):
                    positive_utterance = self.text[self.text["Emotion"]==emotion].sample()
                positive_dialogue_id = positive_utterance["Dialogue_ID"].iloc[0]
                positive_utterance_id = positive_utterance["Utterance_ID"].iloc[0]
                positive = self.get_mel_spectogram(os.path.join(os.path.abspath(self.audio_path), f"dia{positive_dialogue_id}_utt{positive_utterance_id}.wav"))


                negative_utterance = self.text[self.text["Emotion"]!=emotion].sample()
                while  (negative_utterance["Emotion"].iloc[0] == anchor_utterance["Emotion"].iloc[0]):
                    negative_utterance = self.text[self.text["Emotion"]!=emotion].sample()

                negative_dialogue_id = negative_utterance["Dialogue_ID"].iloc[0]
                negative_utterance_id = negative_utterance["Utterance_ID"].iloc[0]
                negative = self.get_mel_spectogram(os.path.join(os.path.abspath(self.audio_path), f"dia{negative_dialogue_id}_utt{negative_utterance_id}.wav"))

                anchor_embedding = model(anchor.unsqueeze(0).to(device))
                positive_embedding = model(positive.unsqueeze(0).to(device))
                negative_embedding = model(negative.unsqueeze(0).to(device))

                anchor_positive_distance = self.compute_distance(anchor_embedding, positive_embedding)
                anchor_negative_distance = self.compute_distance(anchor_embedding, negative_embedding)

                if anchor_positive_distance < anchor_negative_distance < anchor_positive_distance + margin:
                    hard_negative = True
            # print(f"count: {count}")
            if count > 100:
                print("count > 100")


            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)
        return anchors, positives, negatives



    @torch.no_grad()
    def mine_hard_triplets(self, batch_size, model, device):
        embeddings = []
        utterances = []
        anchor = []
        positive = []
        negative = []
        len_triplet_picking = (self.len_triplet_picking // batch_size)
        for _ in range(len_triplet_picking):
            random_audio_mel_spectograms = []
            for _ in range(batch_size):
                emotion = random.choice(list(self.emotion_labels.values()))
                random_utterance = self.text[self.text["Emotion"]==emotion].sample()
                # random_utterance = self.text.sample()

                random_dialogue_id = random_utterance["Dialogue_ID"].iloc[0]
                random_utterance_id = random_utterance["Utterance_ID"].iloc[0]

                utterances.append(random_utterance)

                # Load the audio
                random_wav_path = os.path.join(os.path.abspath(self.audio_path), f"dia{random_dialogue_id}_utt{random_utterance_id}.wav")

                # Get mel spectogram
                random_audio_mel_spectogram = self.get_mel_spectogram(random_wav_path)
                random_audio_mel_spectograms.append(random_audio_mel_spectogram)

                # Compute the embedding
            random_audio_mel_spectograms = torch.stack(random_audio_mel_spectograms).to(device)
            random_embeddings = model(random_audio_mel_spectograms).detach().cpu()
            embeddings.extend(random_embeddings)

        # compute the distance matrix
        embeddings = torch.stack(embeddings).squeeze()
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)

        # POSITIVE
        # get valid mask for positive
        positive_mask = self.compute_positive_mask(utterances)

        # put to 0 the value of distance matrix where the mask is 0
        distance_matrix_positive = distance_matrix * positive_mask

        # get the index of the max value in the distance matrix positive
        positive_index = torch.argmax(distance_matrix_positive, dim=1)

        # NEGATIVE
        # get valid mask for negative
        negative_mask = self.compute_negative_mask(utterances)

        # put to inf the value of distance matrix where the mask is inf
        distance_matrix_negative = distance_matrix + negative_mask

        # get the index of the min value in the distance matrix negative
        negative_index = torch.argmin(distance_matrix_negative, dim=1)

        # BATCH
        losses = torch.tensor( [distance_matrix[index,p] - distance_matrix[index, n] for index,(p,n) in enumerate(zip(positive_index, negative_index))])
        # put to 0 the value if [distance_matrix[index,p] - distance_matrix[index, n] is negative

        # take the batch_size biggest losses
        _, indices = torch.topk(losses, batch_size, sorted=False)

        # get the corresponding utterances of index, p, and n
        index = [utterances[i] for i in indices]
        p = [utterances[i] for i in positive_index[indices]]
        n = [utterances[i] for i in negative_index[indices]]

        # get the corresponding mel spectogram of index, p, and n
        for index_utterance, p_utterance, n_utterance in zip(index, p, n):
            index_dialogue_id = index_utterance["Dialogue_ID"].iloc[0]
            index_utterance_id = index_utterance["Utterance_ID"].iloc[0]
            p_dialogue_id = p_utterance["Dialogue_ID"].iloc[0]
            p_utterance_id = p_utterance["Utterance_ID"].iloc[0]
            n_dialogue_id = n_utterance["Dialogue_ID"].iloc[0]
            n_utterance_id = n_utterance["Utterance_ID"].iloc[0]

            # Load the audio
            index_wav_path = os.path.join(os.path.abspath(self.audio_path), f"dia{index_dialogue_id}_utt{index_utterance_id}.wav")
            p_wav_path = os.path.join(os.path.abspath(self.audio_path), f"dia{p_dialogue_id}_utt{p_utterance_id}.wav")
            n_wav_path = os.path.join(os.path.abspath(self.audio_path), f"dia{n_dialogue_id}_utt{n_utterance_id}.wav")

            # Get mel spectogram
            index_audio_mel_spectogram = self.get_mel_spectogram(index_wav_path)
            p_audio_mel_spectogram = self.get_mel_spectogram(p_wav_path)
            n_audio_mel_spectogram = self.get_mel_spectogram(n_wav_path)

            anchor.append(index_audio_mel_spectogram)
            positive.append(p_audio_mel_spectogram)
            negative.append(n_audio_mel_spectogram)

        anchors = torch.stack(anchor)
        positives = torch.stack(positive)
        negatives = torch.stack(negative)
        return anchors, positives, negatives

    @torch.no_grad()
    def compute_positive_mask (self, utterances):

        n_utterances = len(utterances)
        positive_mask = torch.ones((n_utterances, n_utterances), dtype=torch.float32)
        i_emotions = torch.tensor([utterance["Emotion"].iloc[0] for utterance in utterances])
        j_emotions = i_emotions.unsqueeze(-1)
        positive_mask = torch.where(i_emotions != j_emotions, torch.tensor(0.0), positive_mask)
        positive_mask[torch.eye(n_utterances) == 1] = 0.0

        return positive_mask

    @torch.no_grad()
    def compute_negative_mask (self, utterances):

        n_utterances = len(utterances)
        negative_mask = torch.zeros((n_utterances, n_utterances), dtype=torch.float32)
        i_emotions = torch.tensor([utterance["Emotion"].iloc[0] for utterance in utterances])
        j_emotions = i_emotions.unsqueeze(-1)
        negative_mask = torch.where(i_emotions == j_emotions, float("inf"), negative_mask)
        negative_mask[torch.eye(n_utterances) == 1] = float("inf")

        return negative_mask

    def compute_statistics(self):
            # compute statistics
        max_value = 0
        min_value = float("inf")
        for i in tqdm(range(len(self.text)), desc="Compute statistics"):
            utterance = self.text.iloc[i]
            dialogue_id = utterance["Dialogue_ID"]
            utterance_id = utterance["Utterance_ID"]
            audio_path = os.path.join(os.path.abspath(self.audio_path), f"dia{dialogue_id}_utt{utterance_id}.wav")
            audio, sr = torchaudio.load(audio_path, format="wav", normalize=True)
            audio = torch.nn.functional.pad(audio, (0, self.MAX_AUDIO_LENGTH - audio.shape[1]), mode='constant', value=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_mel_spectogram = torchaudio.transforms.MelSpectrogram (sample_rate = sr, n_fft=400, hop_length=160, n_mels=128)(audio)
                audio_mel_spectogram = torch.tensor(librosa.power_to_db(audio_mel_spectogram))
                if audio_mel_spectogram.max() > max_value:
                    max_value = audio_mel_spectogram.max()
                    # print (max_value)
                if audio_mel_spectogram.min() < min_value:
                    min_value = audio_mel_spectogram.min()
                    # print (min_value)
        return max_value, min_value


if __name__ == "__main__":

    dataset = DatasetMelAudio(mode="train")
    print(dataset[0])

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    # for batch in dataloader:
    #     print(batch)


