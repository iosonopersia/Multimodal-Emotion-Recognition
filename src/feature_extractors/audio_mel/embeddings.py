import os
from munch import Munch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from dataset import Dataset
from AudioMelFeatureExtractor import AudioMelFeatureExtractor
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
        'batch_size': 1,  # We need to preserve the order of the data
        'shuffle': False, # We need to preserve the order of the data
        'num_workers': 0, # We need to preserve the order of the data
        'pin_memory': True
    }
    data_train = Dataset(mode="train", config=config)
    data_val = Dataset(mode="val", config=config)
    data_test = Dataset(mode="test", config=config)

    dl_train = torch.utils.data.DataLoader(data_train, **dataloader_config)
    dl_val = torch.utils.data.DataLoader(data_val, **dataloader_config)
    dl_test = torch.utils.data.DataLoader(data_test, **dataloader_config)

    #============MODEL===============
    #--------------------------------
    model_checkpoint_path = os.path.abspath(config.checkpoint.load_path)
    checkpoint = torch.load(model_checkpoint_path)
    model = AudioMelFeatureExtractor().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    save_path = "embeddings/audio_MEL"
    save_embeddings(dl_train, model, device, save_path, "train")
    save_embeddings(dl_val, model, device, save_path, "val")
    save_embeddings(dl_test, model, device, save_path, "test")

    visualize_model(model, dl_train, device, "2D")
    visualize_model(model, dl_val, device, "2D")
    visualize_model(model, dl_test, device, "2D")


def save_embeddings(dataloader, model, device, path, mode):
    embeddings_list = []

    print(f"Saving {mode} embeddings...")

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader): # TODO: the dataset should give us the indices of each samples, so that we can preserve the order
            mel_spect = batch["audio_mel_spectogram"].to(device)

            embeddings = model(mel_spect)
            embeddings = embeddings.detach().cpu()
            embeddings_list.append(embeddings)

    embeddings_tensor = torch.cat(embeddings_list, dim=0)
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
    predicted_labels = []
    true_labels = []
    with torch.inference_mode():
        for i, data in tqdm(enumerate(dataloader), "Visualizing", total=len(dataloader)):
            inputs, labels = data["audio_mel_spectogram"].to(device), data["emotion"].to(device)
            outputs = model(inputs)
            for j in range(len(inputs)):
                # outputs = tsne.fit_transform(outputs.cpu().detach().numpy())
                embeddings.append(outputs[j].cpu().detach().numpy())
                true_labels.append(labels[j].item())
    # visualize embeddings
    # Extract the x, y, and z coordinates of the 3D embeddings
    # embeddings = tsne.fit_transform(np.array(embeddings))
    embeddings = PCA(random_state=0).fit_transform(np.array(embeddings))[:,:50]
    embeddings = tsne.fit_transform(np.array(embeddings))

    true_labels = np.array(true_labels)
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    if visualization_type == "3D":
        z = embeddings[:, 2]

    # Create a scatter plot of the 3D embeddings
    if visualization_type == "3D":
        fig = px.scatter_3d(x=x, y=y, z=z, text=true_labels, color=true_labels, opacity=0.7, width=800, height=800)
    else:
        fig = px.scatter(x=x, y=y, text=true_labels, color=true_labels, opacity=0.7, width=800, height=800)

    # Show the plot
    fig.show()


if __name__ == "__main__" :
    main()
