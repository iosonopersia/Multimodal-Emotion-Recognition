# from tkinter import Image
import torch
from utils.dataset_utils import get_text
import os
import torchaudio
import matplotlib.pyplot as plt
from torchvision import transforms
import warnings
from PIL import Image
import numpy as np


'''
TODO build function to get the batch, this function will take as input the model, it will select valid triplets until I have
 n of them (with n = 100 in the beginning). A valid triplet is a triplet where the positive has the same label as the anchor and
 the negative has a different label.
 Once I have the valid triplets, I will pass them to the model and I will compute their embeddings and then I will compute the
 distance (just compute the triplet loss) between the embedding and I will choose the B hardest triplets.
'''


class DatasetMelAudio(torch.utils.data.Dataset):
    def __init__(self, mode="train", config=None):
        super().__init__()

        self.MAX_AUDIO_LENGTH = 42000 # 42000 ms = 42 seconds
        self.len_triplet_picking = 100
        self.normalize = transforms.Normalize([0.0,0.0,0.0], [255.0, 255.0, 255.0])
        self.config = config

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
        emotion_labels = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "surprise": 4, "fear": 5, "disgust": 6}
        self.text["Emotion"] = self.text["Emotion"].map(emotion_labels)

        # Count how many dialogues there are
        self.dialogue_ids = self.text["Dialogue_ID"].unique()
        print(f"Loaded {len(self.dialogue_ids)} dialogues for {self.mode}ing")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):

        utterance = self.text.iloc[idx]
        dialogue_id = utterance["Dialogue_ID"]
        utterance_id = utterance["Utterance_ID"]

        if self.mode == "train":
            if (dialogue_id, utterance_id) in {(125, 3)}:
                # This utterance video/audio is corrupted :-(
                return self.__getitem__(idx + 1)
        elif self.mode == "val":
            if (dialogue_id, utterance_id) in {(110, 7)}:
                # This utterance video/audio is corrupted :-(
                return self.__getitem__(idx + 1)
        elif self.mode == "test":
            if (dialogue_id, utterance_id) in {(38,4),(220,0)}:
                # This utterance video/audio is corrupted :-(
                return self.__getitem__(idx + 1)

        # Audio
        wav_path = os.path.join(os.path.abspath(self.audio_path), f"dia{dialogue_id}_utt{utterance_id}.wav")

        # Emotion
        emotion = utterance["Emotion"]
        emotion = torch.tensor([emotion])

        #TODO AUGMENTATION (the audio signal is first processed via different augmentation techniques like time warping and Additive White Gaussian Noise (AWGN) noise)

        audio_mel_spectogram = self.get_mel_spectogram(wav_path)

        return {"audio_mel_spectogram": audio_mel_spectogram, "emotion": emotion}


    def get_labels(self):
        return self.text["Sentiment"].to_numpy(), self.text["Emotion"].to_numpy()

    def get_mel_spectogram(self, audio_path):
        '''
        transform audio into 2D Mel Spectrogram (RGB) :
            the Short Time Fourier transform (STFT) is used with the frame length of 400 samples (25 ms) and hop length of
            160 samples (10ms).
            We also use 128 Mel filter banks to generate the Mel Spectrogram

        '''
        # take last part of the audio_path
        audio_path_cache = audio_path.split("/")[-1]
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

            # plt.imshow( audio_mel_spectogram )
            # audio_mel_spectogram = Image.open(audio_path_cache)
            audio_mel_spectogram = audio_mel_spectogram.permute(2, 0, 1)
            return audio_mel_spectogram

        audio, sr = torchaudio.load(audio_path, format="wav")
        audio = torch.nn.functional.pad(audio, (0, self.MAX_AUDIO_LENGTH - audio.shape[1]), mode='constant', value=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio_mel_spectogram = torchaudio.transforms.MelSpectrogram (sample_rate = sr, n_fft=400, hop_length=160, n_mels=128)(audio)
        # transform to RGB image
        audio_mel_spectogram = audio_mel_spectogram.repeat(3, 1, 1)


        # save to cache
        audio_mel_spectogram_save = audio_mel_spectogram.permute(1, 2, 0) * 255
        audio_mel_spectogram_save = audio_mel_spectogram_save.numpy().astype(np.uint8)
        audio_mel_spectogram_save = Image.fromarray(audio_mel_spectogram_save)
        audio_mel_spectogram_save.save(audio_path_cache)

        # plot
        # plt.imshow( audio_mel_spectogram.permute(1, 2, 0) )

        return audio_mel_spectogram

    def compute_distance(self, anchor, sample):
        distance = torch.norm(anchor - sample)
        return distance

    @torch.no_grad()
    def get_batched_triplets(self, batch_size, model):
        # Take randomly n samples from the dataset
        if self.mode == "val" or self.mode == "test":
            anchors = []
            positives = []
            negatives = []
            # take batch_size valid triples
            for i in range(batch_size):
                #anchors
                anchor_utterance = self.text.sample()
                while (self.check_valid(anchor_utterance) == False):
                    anchor_utterance = self.text.sample()
                anchor_dialogue_id = anchor_utterance["Dialogue_ID"].iloc[0]
                anchor_utterance_id = anchor_utterance["Utterance_ID"].iloc[0]
                anchor = self.get_mel_spectogram(os.path.join(os.path.abspath(self.audio_path), f"dia{anchor_dialogue_id}_utt{anchor_utterance_id}.wav"))
                anchors.append(anchor)

                # positives
                positive_utterance = self.text.sample()
                while (self.check_valid(positive_utterance) == False) or (positive_utterance["Emotion"].iloc[0] != anchor_utterance["Emotion"].iloc[0]):
                    positive_utterance = self.text.sample()
                positive_dialogue_id = positive_utterance["Dialogue_ID"].iloc[0]
                positive_utterance_id = positive_utterance["Utterance_ID"].iloc[0]
                positive = self.get_mel_spectogram(os.path.join(os.path.abspath(self.audio_path), f"dia{positive_dialogue_id}_utt{positive_utterance_id}.wav"))
                positives.append(positive)

                # negatives
                negative_utterance = self.text.sample()
                while (self.check_valid(negative_utterance) == False) or (negative_utterance["Emotion"].iloc[0] == anchor_utterance["Emotion"].iloc[0]):
                    negative_utterance = self.text.sample()
                negative_dialogue_id = negative_utterance["Dialogue_ID"].iloc[0]
                negative_utterance_id = negative_utterance["Utterance_ID"].iloc[0]
                negative = self.get_mel_spectogram(os.path.join(os.path.abspath(self.audio_path), f"dia{negative_dialogue_id}_utt{negative_utterance_id}.wav"))
                negatives.append(negative)
            anchors = torch.stack(anchors)
            positives = torch.stack(positives)
            negatives = torch.stack(negatives)
            return {"anchor": anchors, "positive": positives, "negative": negatives}


        i = 0
        embeddings = []
        utterances = []
        anchor = []
        positive = []
        negative = []
        random_audio_mel_spectograms = []
        device = next(model.parameters()).device
        self.len_triplet_picking = (self.len_triplet_picking // batch_size) * batch_size
        while i < self.len_triplet_picking:
            random_utterance = self.text.sample()
            random_dialogue_id = random_utterance["Dialogue_ID"].iloc[0]
            random_utterance_id = random_utterance["Utterance_ID"].iloc[0]
            if self.check_valid(random_utterance)==False:
                continue
            i+=1
            utterances.append(random_utterance)

            # Load the audio
            random_wav_path = os.path.join(os.path.abspath(self.audio_path), f"dia{random_dialogue_id}_utt{random_utterance_id}.wav")

            # Get mel spectogram
            random_audio_mel_spectogram = self.get_mel_spectogram(random_wav_path)

            random_audio_mel_spectograms.append(random_audio_mel_spectogram)

            # Compute the embedding
            if i % batch_size == 0 and i != 0:
                random_audio_mel_spectograms = torch.stack(random_audio_mel_spectograms).to(device)
                random_embeddings = model(random_audio_mel_spectograms).detach().cpu()
                for random_embedding in random_embeddings:
                    embeddings.append(random_embedding)
                random_audio_mel_spectograms = []

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
        losses = torch.tensor([torch.abs(distance_matrix[index,p] - distance_matrix[index, n]) for index,(p,n) in enumerate(zip(positive_index, negative_index))])

        # take the batch_size biggest losses
        losses, indices = torch.topk(losses, batch_size)

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

        anchor = torch.stack(anchor)
        positive = torch.stack(positive)
        negative = torch.stack(negative)

        return {"anchor": anchor, "positive": positive, "negative": negative}


    def compute_positive_mask (self, utterances):

        positive_mask = torch.ones((len(utterances), len(utterances)))

        for i, utterance in enumerate(utterances):
            positive_mask[i, i] = 0
            for j, other_utterance in enumerate(utterances):
                # if i and j have the same emotion label the put the value to 0 in the mask
                if utterance["Emotion"].iloc[0] != other_utterance["Emotion"].iloc[0]:
                    positive_mask[i, j] = 0

        return positive_mask

    def compute_negative_mask (self, utterances):
        inf = torch.tensor(float("inf"))
        negative_mask = torch.zeros((len(utterances), len(utterances)))

        for i, utterance in enumerate(utterances):
            negative_mask[i, i] = inf # put inf to the diagonal

            for j, other_utterance in enumerate(utterances):
                # if i and j have the same emotion label the put the value to inf in the mask
                if utterance["Emotion"].iloc[0] == other_utterance["Emotion"].iloc[0]:
                    negative_mask[i, j] = inf

        return negative_mask

    def check_valid (self, utterance):
        dialogue_id = utterance["Dialogue_ID"].iloc[0]
        utterance_id = utterance["Utterance_ID"].iloc[0]
        if self.mode == "train":
            if (dialogue_id, utterance_id) in {(125, 3)}:
                # This utterance video/audio is corrupted :-(
                return False
        elif self.mode == "val":
            if (dialogue_id, utterance_id) in {(110, 7)}:
                # This utterance video/audio is corrupted :-(
                return False
        elif self.mode == "test":
            if (dialogue_id, utterance_id) in {(38,4),(220,0)}:
                # This utterance video/audio is corrupted :-(
                return False
        return True






if __name__ == "__main__":

    dataset = DatasetMelAudio(mode="train")
    print(dataset[0])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch)


