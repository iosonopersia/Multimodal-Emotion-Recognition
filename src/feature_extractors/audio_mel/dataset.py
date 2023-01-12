import random
import torch
from utils import get_text
import os
import torchaudio
from PIL import Image
import numpy as np
import librosa
import audiomentations
from audiomentations import AddGaussianSNR, TimeStretch, PitchShift, Shift
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode="train", config=None, generate_mel_spectrograms=False):
        super().__init__()

        self.config = config
        self.MAX_AUDIO_LENGTH = config.AUDIO.max_duration # seconds
        self.len_triplet_picking = config.solver.len_triplet_picking
        self.mode = mode

        # add audio augmentation
        self.audio_transforms = audiomentations.Compose([
            AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=40.0, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ])

        self.augmentation_factor = config.AUDIO.augmentation_factor
        if self.augmentation_factor <= 0:
            self.augmentation_factor = 1

        if self.mode == "train":
            self.audio_path = "data/MELD.Raw/train_splits/wav"
            self.mel_spectrogram_cache = "data/MELD.Raw/train_splits/mel_spectrograms"
            self.augmentation_cache = "data/MELD.Raw/train_splits/augmentation"
            self.augmentation_cache = os.path.abspath(self.augmentation_cache)
            os.makedirs(self.augmentation_cache, exist_ok=True)
        elif self.mode == "val":
            self.audio_path = "data/MELD.Raw/dev_splits_complete/wav"
            self.mel_spectrogram_cache = "data/MELD.Raw/dev_splits_complete/mel_spectrograms"
        elif self.mode == "test":
            self.audio_path = "data/MELD.Raw/output_repeated_splits_test/wav"
            self.mel_spectrogram_cache = "data/MELD.Raw/output_repeated_splits_test/mel_spectrograms"
        else:
            raise ValueError(f"Invalid mode {mode}")
        self.audio_path = os.path.abspath(self.audio_path)
        self.mel_spectrogram_cache = os.path.abspath(self.mel_spectrogram_cache)
        os.makedirs(self.mel_spectrogram_cache, exist_ok=True)

        self.text = get_text(mode)
        # ONLY FOR DEBUGGING
        if self.config.DEBUG.enabled == True:
            self.text = self.text.iloc[0:self.config.DEBUG.num_samples]

        # Map labels to class indices
        self.emotion_labels = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "surprise": 4, "fear": 5, "disgust": 6}
        self.text["Emotion"] = self.text["Emotion"].map(self.emotion_labels)

        # Count how many dialogues there are
        self.dialogue_ids = self.text["Dialogue_ID"].unique()
        print(f"Loaded {len(self.dialogue_ids)} dialogues for {self.mode}ing")

        # Generate mel spectrograms beforehand
        if generate_mel_spectrograms:
            self._generate_all_mel_spectograms()


    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        utterance = self.text.iloc[idx]
        dialogue_id = utterance["Dialogue_ID"]
        utterance_id = utterance["Utterance_ID"]

        # Audio
        wav_path = os.path.join(self.audio_path, f"dia{dialogue_id}_utt{utterance_id}.wav")
        audio_mel_spectogram = self.get_mel_spectrogram(wav_path, augment=False)

        # Emotion
        emotion = utterance["Emotion"]
        emotion = torch.tensor([emotion])

        return {"idx": idx, "audio_mel_spectogram": audio_mel_spectogram, "emotion": emotion}

    def get_labels(self):
        return self.text["Emotion"].to_numpy()

    # from https://github.com/liuxubo717/cl4ac
    def _get_mel_spectrogram(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        y = audio_data/abs(audio_data).max()
        mel_bands = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=400, hop_length=160, win_length=400,
            window='hann', center=True, power=1, n_mels=128,
            fmin=0, fmax=None, htk=False, norm=1).T

        return np.log(mel_bands + np.finfo(float).eps)

    def _add_to_cache(self, mel_spectrogram, cache_path):
        mel_spectrogram = mel_spectrogram * 255.0
        mel_spectrogram = mel_spectrogram.numpy().astype(np.uint8)

        mel_spectrogram = Image.fromarray(mel_spectrogram.squeeze(), mode="L")
        mel_spectrogram.save(cache_path, mode="L")

    def _get_from_cache(self, cache_path):
        mel_spectrogram = Image.open(cache_path)
        mel_spectrogram = np.array(mel_spectrogram, dtype=np.float32)
        mel_spectrogram = torch.from_numpy(mel_spectrogram) / 255.0
        mel_spectrogram = mel_spectrogram.unsqueeze(dim=-1)

        return mel_spectrogram

    def get_mel_spectrogram(self, audio_path, augment=True):
        '''
        transform audio into 2D Mel Spectrogram (RGB) :
            the Short Time Fourier transform (STFT) is used with the frame length of 400 samples (25 ms) and hop length of
            160 samples (10ms).
            We also use 128 Mel filter banks to generate the Mel Spectrogram
        '''
        #choose whether to use augmentation or not
        if self.mode == "train" and augment==True:
            augment = random.randint (0, self.augmentation_factor - 1)
        else:
            augment = 0

        if augment == 0:
            cache_path = os.path.basename(audio_path)
            cache_path = cache_path.split(".")[0]
            cache_path = os.path.join(self.mel_spectrogram_cache, f"{cache_path}.png")
        else:
            cache_path = os.path.basename(audio_path)
            cache_path = cache_path.split(".")[0]
            cache_path = os.path.join(self.augmentation_cache, f"{cache_path}_{augment}.png")

        sr = self.config.AUDIO.ffmpeg_sr

        if os.path.exists(cache_path):
            # Cache hit
            mel_spectrogram = self._get_from_cache(cache_path)
        else:
            # Cache miss
            audio, _sr = torchaudio.load(audio_path, format="wav", normalize=True)
            if _sr != sr:
                raise ValueError(f"Sample rate mismatch: {_sr} != {sr}")

            # Truncate waveform to max length in seconds
            max_audio_length = int(self.MAX_AUDIO_LENGTH * sr)
            if audio.shape[-1] > max_audio_length:
                audio = audio[..., :max_audio_length]

            audio = audio.numpy()
            if augment > 0:
                audio = self.audio_transforms(audio, sample_rate=sr)

            mel_spectrogram = self._get_mel_spectrogram(audio, sr)
            mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

            min_intensity = mel_spectrogram.min()
            max_intensity = mel_spectrogram.max()
            mel_spectrogram = (mel_spectrogram - min_intensity) / (max_intensity - min_intensity)

            self._add_to_cache(mel_spectrogram, cache_path)

        # Move channel dimension to first dimension
        mel_spectrogram = mel_spectrogram.permute(2, 0, 1)
        # Add padding if necessary
        max_spectrogram_rows = int(self.MAX_AUDIO_LENGTH * (sr / 160.0)) + 1 # hop_length = 160
        mel_spectrogram = torch.nn.functional.pad(
            mel_spectrogram,
            (0, 0, 0, max_spectrogram_rows - mel_spectrogram.shape[1]),
            mode='constant',
            value=0.0)
        # Convert to RGB
        mel_spectrogram = mel_spectrogram.repeat(3, 1, 1)

        return mel_spectrogram

    def compute_distance(self, anchor, sample):
        distance = torch.norm(anchor - sample, p=2)
        return distance

    @torch.no_grad()
    def get_batched_triplets(self, batch_size, model, mining_type="random", margin=1, device='cpu'):
        model.eval()

        if mining_type == "random":
            anchors, positives, negatives = self.mine_random_triplets(batch_size)
        elif mining_type == "semi-hard":
            anchors, positives, negatives = self.mine_semihard_triplets(batch_size, model, device, margin)
        elif mining_type == "hard":
            anchors, positives, negatives = self.mine_hard_triplets(batch_size, model, device)
        else:
            raise ValueError("mining_type must be 'hard', 'semi-hard' or 'random'")

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
            anchor = self.get_mel_spectrogram(os.path.join(self.audio_path, f"dia{anchor_dialogue_id}_utt{anchor_utterance_id}.wav"))
            anchors.append(anchor)

            # positives
            positive_utterance = self.text[self.text["Emotion"]==emotion].sample()
            while ((positive_utterance["Emotion"].iloc[0] != anchor_utterance["Emotion"].iloc[0]) or (positive_utterance["Dialogue_ID"].iloc[0]  == anchor_utterance["Dialogue_ID"].iloc[0]  and positive_utterance["Utterance_ID"].iloc[0]  == anchor_utterance["Utterance_ID"].iloc[0] )):
                positive_utterance = self.text[self.text["Emotion"]==emotion].sample()
            positive_dialogue_id = positive_utterance["Dialogue_ID"].iloc[0]
            positive_utterance_id = positive_utterance["Utterance_ID"].iloc[0]
            positive = self.get_mel_spectrogram(os.path.join(self.audio_path, f"dia{positive_dialogue_id}_utt{positive_utterance_id}.wav"))
            positives.append(positive)

            # negatives
            negative_utterance = self.text[self.text["Emotion"]!=emotion].sample()
            while (negative_utterance["Emotion"].iloc[0] == anchor_utterance["Emotion"].iloc[0]):
                negative_utterance = self.text[self.text["Emotion"]!=emotion].sample()
            negative_dialogue_id = negative_utterance["Dialogue_ID"].iloc[0]
            negative_utterance_id = negative_utterance["Utterance_ID"].iloc[0]
            negative = self.get_mel_spectrogram(os.path.join(self.audio_path, f"dia{negative_dialogue_id}_utt{negative_utterance_id}.wav"))
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
                anchor = self.get_mel_spectrogram(os.path.join(self.audio_path, f"dia{anchor_dialogue_id}_utt{anchor_utterance_id}.wav"))

                # positives
                positive_utterance = self.text[self.text["Emotion"]==emotion].sample()
                while (positive_utterance["Emotion"].iloc[0] != anchor_utterance["Emotion"].iloc[0]) or (positive_utterance["Dialogue_ID"].iloc[0] == anchor_utterance["Dialogue_ID"].iloc[0]  and positive_utterance["Utterance_ID"].iloc[0]  == anchor_utterance["Utterance_ID"].iloc[0] ):
                    positive_utterance = self.text[self.text["Emotion"]==emotion].sample()
                positive_dialogue_id = positive_utterance["Dialogue_ID"].iloc[0]
                positive_utterance_id = positive_utterance["Utterance_ID"].iloc[0]
                positive = self.get_mel_spectrogram(os.path.join(self.audio_path, f"dia{positive_dialogue_id}_utt{positive_utterance_id}.wav"))

                negative_utterance = self.text[self.text["Emotion"]!=emotion].sample()
                while  (negative_utterance["Emotion"].iloc[0] == anchor_utterance["Emotion"].iloc[0]):
                    negative_utterance = self.text[self.text["Emotion"]!=emotion].sample()

                negative_dialogue_id = negative_utterance["Dialogue_ID"].iloc[0]
                negative_utterance_id = negative_utterance["Utterance_ID"].iloc[0]
                negative = self.get_mel_spectrogram(os.path.join(self.audio_path, f"dia{negative_dialogue_id}_utt{negative_utterance_id}.wav"))

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

    # @torch.no_grad()
    def mine_hard_triplets(self, batch_size, model, device, criterion=None):

        utterances = []
        anchor = []
        positive = []
        negative = []
        len_triplet_picking = (self.len_triplet_picking // batch_size)
        embeddings = torch.zeros((len_triplet_picking*batch_size, 300)).to(device)
        for i in range(len_triplet_picking):
            random_audio_mel_spectograms = []
            with torch.no_grad():
                for _ in range(batch_size):

                    emotion = random.choice(list(self.emotion_labels.values()))
                    random_utterance = self.text[self.text["Emotion"]==emotion].sample()
                    # random_utterance = self.text.sample()

                    random_dialogue_id = random_utterance["Dialogue_ID"].iloc[0]
                    random_utterance_id = random_utterance["Utterance_ID"].iloc[0]

                    utterances.append(random_utterance)

                    # Load the audio
                    random_wav_path = os.path.join(self.audio_path, f"dia{random_dialogue_id}_utt{random_utterance_id}.wav")

                    # Get mel spectogram
                    random_audio_mel_spectogram = self.get_mel_spectrogram(random_wav_path)
                    random_audio_mel_spectograms.append(random_audio_mel_spectogram)

                        # Compute the embedding
                random_audio_mel_spectograms = torch.stack(random_audio_mel_spectograms).to(device)

            random_embeddings = model(random_audio_mel_spectograms)#.detach().cpu()
            embeddings[i*batch_size:(i+1)*batch_size] = random_embeddings

        # compute the distance matrix
        #embeddings = torch.stack(embeddings).squeeze()
        # if self.mode == "train":
        #     embeddings.requires_grad = True

        distance_matrix = torch.cdist(embeddings, embeddings, p=2)

        # POSITIVE
        # get valid mask for positive
        positive_mask = self.compute_positive_mask(utterances)

        # put to 0 the value of distance matrix where the mask is 0
        distance_matrix_positive = distance_matrix * positive_mask.cuda()

        # get the index of the max value in the distance matrix positive
        positive_index = torch.argmax(distance_matrix_positive, dim=1)

        # NEGATIVE
        # get valid mask for negative
        negative_mask = self.compute_negative_mask(utterances)

        # put to inf the value of distance matrix where the mask is inf
        distance_matrix_negative = distance_matrix + negative_mask.cuda()

        # get the index of the min value in the distance matrix negative
        negative_index = torch.argmin(distance_matrix_negative, dim=1)

        # BATCH
        losses = torch.tensor( [distance_matrix[index,p] - distance_matrix[index, n] for index,(p,n) in enumerate(zip(positive_index, negative_index))])

        # take the batch_size biggest losses
        if self.mode == "train":
            _, indices = torch.topk(losses, (self.len_triplet_picking // batch_size) * batch_size, sorted=False)
        else:
            _, indices = torch.topk(losses, batch_size, sorted=False)

        embeddings_anchors = embeddings[indices]
        embeddings_positives = embeddings[positive_index[indices]]
        embeddings_negatives = embeddings[negative_index[indices]]
        #compute the triplet loss
        loss = criterion(embeddings_anchors,  embeddings_positives, embeddings_negatives)
        return loss
        # # get the corresponding utterances of index, p, and n
        # index = [utterances[i] for i in indices]
        # p = [utterances[i] for i in positive_index[indices]]
        # n = [utterances[i] for i in negative_index[indices]]

        # # get the corresponding mel spectogram of index, p, and n
        # for index_utterance, p_utterance, n_utterance in zip(index, p, n):
        #     index_dialogue_id = index_utterance["Dialogue_ID"].iloc[0]
        #     index_utterance_id = index_utterance["Utterance_ID"].iloc[0]
        #     p_dialogue_id = p_utterance["Dialogue_ID"].iloc[0]
        #     p_utterance_id = p_utterance["Utterance_ID"].iloc[0]
        #     n_dialogue_id = n_utterance["Dialogue_ID"].iloc[0]
        #     n_utterance_id = n_utterance["Utterance_ID"].iloc[0]

        #     # Load the audio
        #     index_wav_path = os.path.join(self.audio_path, f"dia{index_dialogue_id}_utt{index_utterance_id}.wav")
        #     p_wav_path = os.path.join(self.audio_path, f"dia{p_dialogue_id}_utt{p_utterance_id}.wav")
        #     n_wav_path = os.path.join(self.audio_path, f"dia{n_dialogue_id}_utt{n_utterance_id}.wav")

        #     # Get mel spectogram
        #     index_audio_mel_spectogram = self.get_mel_spectrogram(index_wav_path)
        #     p_audio_mel_spectogram = self.get_mel_spectrogram(p_wav_path)
        #     n_audio_mel_spectogram = self.get_mel_spectrogram(n_wav_path)

        #     anchor.append(index_audio_mel_spectogram)
        #     positive.append(p_audio_mel_spectogram)
        #     negative.append(n_audio_mel_spectogram)

        # anchors = torch.stack(anchor)
        # positives = torch.stack(positive)
        # negatives = torch.stack(negative)
        # return anchors, positives, negatives

    # @torch.no_grad()
    def compute_positive_mask (self, utterances):
        n_utterances = len(utterances)
        positive_mask = torch.ones((n_utterances, n_utterances), dtype=torch.float32)
        i_emotions = torch.tensor([utterance["Emotion"].iloc[0] for utterance in utterances])
        j_emotions = i_emotions.unsqueeze(-1)
        positive_mask = torch.where(i_emotions != j_emotions, torch.tensor(0.0), positive_mask)
        positive_mask.diagonal()[:] = 0.0 # The positive cannot coincide with the anchor

        return positive_mask

    # @torch.no_grad()
    def compute_negative_mask (self, utterances):
        n_utterances = len(utterances)
        negative_mask = torch.zeros((n_utterances, n_utterances), dtype=torch.float32)
        i_emotions = torch.tensor([utterance["Emotion"].iloc[0] for utterance in utterances])
        j_emotions = i_emotions.unsqueeze(-1)
        negative_mask = torch.where(i_emotions == j_emotions, float("inf"), negative_mask)
        negative_mask.diagonal()[:] = float("inf") # The negative cannot coincide with the anchor

        return negative_mask

    def _generate_all_mel_spectograms(self):
        sr = self.config.AUDIO.ffmpeg_sr
        for _, utterance in tqdm(self.text.iterrows(), "Generating mel spectograms", total=len(self.text)):
            dialogue_id = utterance["Dialogue_ID"]
            utterance_id = utterance["Utterance_ID"]
            audio_path = os.path.join(self.audio_path, f"dia{dialogue_id}_utt{utterance_id}.wav")
            audio, _sr = torchaudio.load(audio_path, format="wav", normalize=True)
            if _sr != sr:
                raise ValueError(f"Sample rate mismatch: {_sr} != {sr}")

            # Truncate waveform to max length in seconds
            max_audio_length = int(self.MAX_AUDIO_LENGTH * sr)
            if audio.shape[-1] > max_audio_length:
                audio = audio[..., :max_audio_length]

            audio = audio.numpy()
            if self.mode == "train":
                for augment in range(self.augmentation_factor):
                    if augment == 0:
                        cache_path = os.path.basename(audio_path)
                        cache_path = cache_path.split(".")[0]
                        cache_path = os.path.join(self.mel_spectrogram_cache, f"{cache_path}.png")
                    else:
                        cache_path = os.path.basename(audio_path)
                        cache_path = cache_path.split(".")[0]
                        cache_path = os.path.join(self.augmentation_cache, f"{cache_path}_{augment}.png")

                    if augment > 0:
                        audio = self.audio_transforms(audio, sample_rate=sr)

                    mel_spectrogram = self._get_mel_spectrogram(audio, sr)
                    mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

                    min_intensity = mel_spectrogram.min()
                    max_intensity = mel_spectrogram.max()
                    mel_spectrogram = (mel_spectrogram - min_intensity) / (max_intensity - min_intensity)

                    self._add_to_cache(mel_spectrogram, cache_path)
            else:
                cache_path = os.path.basename(audio_path)
                cache_path = cache_path.split(".")[0]
                cache_path = os.path.join(self.mel_spectrogram_cache, f"{cache_path}.png")
                mel_spectrogram = self._get_mel_spectrogram(audio, sr)
                mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

                min_intensity = mel_spectrogram.min()
                max_intensity = mel_spectrogram.max()
                mel_spectrogram = (mel_spectrogram - min_intensity) / (max_intensity - min_intensity)

                self._add_to_cache(mel_spectrogram, cache_path)


