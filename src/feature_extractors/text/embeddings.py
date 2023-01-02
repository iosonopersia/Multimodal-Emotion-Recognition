import os
import torch
from utils import get_config
from dataset import Dataset
from tqdm import tqdm
from transformers import RobertaModel
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
        'batch_size': 1, # We need to preserve the order of the data
        'shuffle': False, # We need to preserve the order of the data
        'num_workers': 0, # We need to preserve the order of the data
        'pin_memory': True
    }
    data_train = Dataset(mode="train")
    data_val = Dataset(mode="val")
    data_test = Dataset(mode="test")

    dl_train = torch.utils.data.DataLoader(data_train, collate_fn=data_train.collate_fn, **dataloader_config)
    dl_val = torch.utils.data.DataLoader(data_val, collate_fn=data_val.collate_fn, **dataloader_config)
    dl_test = torch.utils.data.DataLoader(data_test, collate_fn=data_test.collate_fn, **dataloader_config)

    #============MODEL===============
    #--------------------------------
    roberta_checkpoint_path = os.path.abspath(config.save_pretrained.path)
    model = RobertaModel.from_pretrained(roberta_checkpoint_path, add_pooling_layer=False).to(device)

    save_path = "embeddings/text"
    save_embeddings(dl_train, model, device, save_path, "train")
    save_embeddings(dl_val, model, device, save_path, "val")
    save_embeddings(dl_test, model, device, save_path, "test")


def save_embeddings(dataloader, model, device, path, mode):
    embeddings_list = []

    print(f"Saving {mode} embeddings...")

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader): # TODO: the dataset should give us the indices of each samples, so that we can preserve the order
            text, emotion, attention_mask = batch["text"], batch["emotion"], batch["attention_mask"]

            text = text.to(device)
            emotion = emotion.to(device)
            attention_mask = attention_mask.to(device)

            embeddings = model(text, attention_mask)
            embeddings = embeddings.last_hidden_state
            embeddings = embeddings[:, 0, :] # [CLS] token
            embeddings = embeddings.detach().cpu()
            embeddings_list.append(embeddings)

    embeddings_tensor = torch.cat(embeddings_list, dim=0)
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(os.path.abspath(path), f"{mode}.pkl")
    pickle.dump(embeddings_tensor, open(save_path, "wb"))

    print(f"Saved {mode} embeddings to {save_path}")
    print()


if __name__ == "__main__" :
    main()
