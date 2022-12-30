import os
import torch
from utils import get_config
from datasets.dataset import Dataset
from models.FeatureExtractor import FeatureExtractor
from models.M2FNet import M2FNet
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Suppress warnings from 'transformers' package
from transformers import logging
logging.set_verbosity_error()


def main(config=None):
    #CONFIG
    config = get_config()

    #============DEVICE===============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}...")

    #============LOAD DATA===============
    #------------------------------------
    # TEST DATA
    data_test = Dataset(mode="test")

    test_dl_cfg = config.test.data_loader
    dl_test = torch.utils.data.DataLoader(data_test, collate_fn=data_test.my_collate_fn, **test_dl_cfg)

    #============MODEL===============
    #--------------------------------
    feature_embedding_model = FeatureExtractor().to(device)
    model = M2FNet(config.model).to(device)

    #============LOAD================
    #--------------------------------
    load_checkpoint_path = config.checkpoint.load_path

    if (os.path.exists(load_checkpoint_path)):
        checkpoint = torch.load(load_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("Checkpoint not found")

    #============TRAIN===============
    #--------------------------------
    print("Testing...")
    accuracy, f1 = test(model, feature_embedding_model, dl_test, device)
    print (f"Accuracy: {accuracy} F1: {f1}")
    print("Testing complete")


def test(model, feature_embedding_model, dl_test, device):
    accuracy = 0
    f1 = 0
    feature_embedding_model.eval()
    model.eval()
    with torch.inference_mode():
        for data in tqdm(dl_test, total=len(dl_test)):
            text, audio, emotion = data["text"], data["audio"], data["emotion"]
            emotion = emotion.to(device)

            text = [t.to(device) for t in text]
            audio = [[aa.to(device) for aa in a] for a in audio]
            text, audio, mask = feature_embedding_model(text, audio)

            outputs = model(text, audio, mask)
            emotion_predicted = torch.argmax(outputs, dim=2)
            mask = (emotion != -1)
            emotion_predicted = emotion_predicted[mask]
            emotion = emotion[mask]
            accuracy += accuracy_score(emotion.flatten().detach().cpu().numpy(), emotion_predicted.flatten().detach().cpu().numpy())
            f1 += f1_score(emotion.flatten().detach().cpu().numpy(), emotion_predicted.flatten().detach().cpu().numpy(), average='weighted')

    return accuracy/len(dl_test), f1/len(dl_test)


if __name__ == "__main__":
    main()
