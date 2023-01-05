import os
import torch
from utils import get_config
from dataset import Dataset, collate_fn
from tqdm import tqdm
from torchaudio.pipelines import WAV2VEC2_BASE as WAV2VEC2
import pickle

# Suppress warnings from 'transformers' package
from transformers import logging
logging.set_verbosity_error()


def main():
    #CONFIG
    config = get_config()

    #============DEVICE===============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}...")

    #============LOAD DATA===============
    #------------------------------------
    dataloader_config = {
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 2,
        'pin_memory': True,
        'drop_last': False
    }
    data_train = Dataset(mode="train")
    data_val = Dataset(mode="val")
    data_test = Dataset(mode="test")

    dl_train = torch.utils.data.DataLoader(data_train, collate_fn=collate_fn, **dataloader_config)
    dl_val = torch.utils.data.DataLoader(data_val, collate_fn=collate_fn, **dataloader_config)
    dl_test = torch.utils.data.DataLoader(data_test, collate_fn=collate_fn, **dataloader_config)

    #============MODEL===============
    #--------------------------------
    model = WAV2VEC2.get_model()
    checkpoint_path = os.path.abspath(config.checkpoint.save_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        wav2vec2_prefix = "wav2vec2."
        wav2vec2_state_dict = {
            key.removeprefix(wav2vec2_prefix): value
            for (key,value) in checkpoint["model_state_dict"].items()
            if key.startswith(wav2vec2_prefix)
        }
        model.load_state_dict(wav2vec2_state_dict)
    else:
        raise ValueError("Checkpoint not found")
    model = model.to(device)

    save_path = "embeddings/audio_wav2vec2"
    save_embeddings(dl_train, model, device, save_path, "train")
    save_embeddings(dl_val, model, device, save_path, "val")
    save_embeddings(dl_test, model, device, save_path, "test")


def save_embeddings(dataloader, model, device, path, mode):
    embeddings_tensor = torch.zeros(len(dataloader.dataset), 768, dtype=torch.float32)

    print(f"Saving {mode} embeddings...")

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            idx = batch["idx"]
            audio = batch["audio"].to(device)
            lengths = batch["lengths"].to(device)

            embeddings = model(audio, lengths)
            embeddings = embeddings[0] # embeddings is a tuple of (hidden_states, lengths)
            embeddings = embeddings[:, 0, :] # [CLS] token
            embeddings = embeddings.cpu()

            embeddings_tensor[idx] = embeddings

    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(os.path.abspath(path), f"{mode}.pkl")
    pickle.dump(embeddings_tensor, open(save_path, "wb"))

    print(f"Saved {mode} embeddings to {save_path}")
    print()


if __name__ == "__main__" :
    main()
