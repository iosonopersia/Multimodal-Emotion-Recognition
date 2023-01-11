import os
from munch import Munch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from dataset import Dataset
from model import AudioMelFeatureExtractor
from tqdm import tqdm
import pickle
import numpy as np
import plotly.express as px


def main():
    #CONFIG
    with open('./src/feature_extractors/audio_mel/config_audio_mel.yaml', 'rt', encoding='utf-8') as f:
        config = Munch.fromYAML(f.read())

    #============DEVICE===============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}...")

    #============LOAD DATA===============
    #------------------------------------
    dataloader_config = {
        'batch_size': 128,
        'shuffle': False,
        'num_workers': 2,
        'pin_memory': True,
        'drop_last': False
    }
    data_train = Dataset(mode="train", config=config)
    data_val = Dataset(mode="val", config=config)
    data_test = Dataset(mode="test", config=config)

    dl_train = torch.utils.data.DataLoader(data_train, **dataloader_config)
    dl_val = torch.utils.data.DataLoader(data_val, **dataloader_config)
    dl_test = torch.utils.data.DataLoader(data_test, **dataloader_config)

    #============MODEL===============
    #--------------------------------
    model = AudioMelFeatureExtractor()
    checkpoint_path = os.path.abspath(config.checkpoint.load_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("Checkpoint not found")
    model = model.to(device)

    save_path = "embeddings/audio_mel"
    save_embeddings(dl_train, model, device, save_path, "train")
    save_embeddings(dl_val, model, device, save_path, "val")
    save_embeddings(dl_test, model, device, save_path, "test")

    visualize_model(model, dl_train, device, "3D")
    visualize_model(model, dl_val, device, "3D")
    visualize_model(model, dl_test, device, "3D")


def save_embeddings(dataloader, model, device, path, mode):
    embeddings_tensor = torch.zeros(len(dataloader.dataset), 300, dtype=torch.float32)

    print(f"Saving {mode} embeddings...")

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            idx = batch["idx"]
            mel_spect = batch["audio_mel_spectogram"].to(device)

            embeddings = model(mel_spect)
            embeddings_tensor[idx] = embeddings.cpu()

    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(os.path.abspath(path), f"{mode}.pkl")
    pickle.dump(embeddings_tensor, open(save_path, "wb"))

    print(f"Saved {mode} embeddings to {save_path}")
    print()


def visualize_model(model, dataloader, device, visualization_type= "3D"):
    if visualization_type == "3D":
        tsne = TSNE(n_components=3)
    elif visualization_type == "2D":
        tsne = TSNE(n_components=2)
    else:
        raise ValueError("Visualization type not supported")

    model.eval()
    embeddings = []
    true_labels = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Visualizing", total=len(dataloader)):
            inputs = batch["audio_mel_spectogram"].to(device)
            outputs = model(inputs)
            embeddings.extend(outputs.cpu().tolist())
            true_labels.extend(batch["emotion"].squeeze(dim=-1).tolist())
    embeddings = np.array(embeddings, dtype=np.float32)
    true_labels = np.array(true_labels, dtype=np.int32)

    # visualize embeddings
    embeddings = PCA(random_state=0).fit_transform(embeddings)[:,:50]
    embeddings = tsne.fit_transform(embeddings)

    x = embeddings[:, 0]
    y = embeddings[:, 1]
    if visualization_type == "3D":
        z = embeddings[:, 2]

    # Create a scatter plot of the 3D embeddings
    if visualization_type == "3D":
        fig = px.scatter_3d(x=x, y=y, z=z, color=true_labels, opacity=0.7, width=800, height=800)
    else:
        fig = px.scatter(x=x, y=y, color=true_labels, opacity=0.7, width=800, height=800)

    # Disable hover data
    fig.update_traces(hovertemplate=None, hoverinfo="skip")

    # Show the plot
    fig.show()


if __name__ == "__main__" :
    main()
