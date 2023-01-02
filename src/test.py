import os
import torch
from utils import get_config
from dataset import Dataset
from model import M2FNet
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
    dl_test = torch.utils.data.DataLoader(data_test, collate_fn=data_test.collate_fn, **test_dl_cfg)

    #============MODEL===============
    #--------------------------------
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
    accuracy, f1 = test(model, dl_test, device)
    print (f"Accuracy: {accuracy} F1: {f1}")
    print("Testing complete")


def test(model, dl_test, device):
    accuracy = 0
    f1 = 0

    model.eval()
    with torch.inference_mode():
        for data in tqdm(dl_test, total=len(dl_test)):
            text = data["text"].to(device)
            audio = data["audio"].to(device)
            emotion = data["emotion"].to(device)
            padding_mask = data["padding_mask"].to(device)

            outputs = model(text, audio, padding_mask)

            emotion_predicted = torch.argmax(outputs, dim=2)
            mask = (emotion != -1)
            emotion_predicted = emotion_predicted[mask].flatten().cpu().numpy()
            emotion = emotion[mask].flatten().cpu().numpy()

            accuracy += accuracy_score(emotion, emotion_predicted)
            f1 += f1_score(emotion, emotion_predicted, average='weighted')

    return accuracy/len(dl_test), f1/len(dl_test)


if __name__ == "__main__":
    main()
