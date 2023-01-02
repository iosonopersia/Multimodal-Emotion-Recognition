import os
import torch
from utils import get_config
from dataset import Dataset
from model import TextERC
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
    model = TextERC().to(device)

    #============LOAD================
    #--------------------------------
    load_checkpoint_path = os.path.abspath(config.checkpoint.save_path)

    if (os.path.exists(load_checkpoint_path)):
        checkpoint = torch.load(load_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("Checkpoint not found")

    #============TRAIN===============
    #--------------------------------
    print("Testing...")
    accuracy, weighted_f1 = test(model, dl_test, device)
    print (f"Accuracy=[{accuracy * 100:.3f}%] Weighted_F1=[{weighted_f1 * 100:.3f}%]")
    print("Testing complete")


def test(model, dl_test, device):
    accuracy = 0
    weighted_f1 = 0
    model.eval()
    with torch.inference_mode():
        for data in tqdm(dl_test, total=len(dl_test)):
            text, emotion, attention_mask = data["text"], data["emotion"], data["attention_mask"]

            text = text.to(device)
            emotion = emotion.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(text, attention_mask)

            # Calculate metrics
            emotion_predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            emotion = emotion.cpu().numpy()
            accuracy += accuracy_score(emotion, emotion_predicted)
            weighted_f1 += f1_score(emotion, emotion_predicted, average='weighted')

    return accuracy/len(dl_test), weighted_f1/len(dl_test)


if __name__ == "__main__":
    main()
