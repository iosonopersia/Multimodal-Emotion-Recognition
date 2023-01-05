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


def extract_wav2vec2_state_dict(checkpoint):
    wav2vec2_prefix = "wav2vec2."
    wav2vec2_state_dict = {
            key.removeprefix(wav2vec2_prefix): value
            for (key,value) in checkpoint["model_state_dict"].items()
            if key.startswith(wav2vec2_prefix)
        }

    return wav2vec2_state_dict


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
        wav2vec2_state_dict = extract_wav2vec2_state_dict(checkpoint)
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

            out = model(audio, lengths)
            hidden_states = out[0] # (batch_size, seq_len, 768)
            lengths = out[1] # (batch_size,)

            # Mean pooling over non-padded elements:
            for i, length in enumerate(lengths):
                hidden_states[i, length:, :] = 0
            embeddings = torch.sum(hidden_states, dim=1) / lengths.unsqueeze(dim=1).type(torch.float32)

            embeddings_tensor[idx] = embeddings.cpu()

    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(os.path.abspath(path), f"{mode}.pkl")
    pickle.dump(embeddings_tensor, open(save_path, "wb"))

    print(f"Saved {mode} embeddings to {save_path}")
    print()


if __name__ == "__main__" :
    main()
