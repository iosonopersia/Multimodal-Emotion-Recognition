import os
import pandas as pd


def get_text (mode="train"):
    """Returns the transcripts for the given mode (train, val, test)."""
    assert mode in ["train", "val", "test"]
    root = os.path.join("data", "MELD.Raw")

    if mode == "train":
        data_path = os.path.join(root, "train_sent_emo.csv")
    elif mode == "val":
        data_path = os.path.join(root, "dev_sent_emo.csv")
    elif mode == 'test':
        data_path = os.path.join(root, "test_sent_emo.csv")

    if not os.path.exists(data_path):
        raise ValueError("Dataset not found at {}".format(data_path))

    df = pd.read_csv(data_path)

    return df


if __name__ == "__main__":

    # Test get_text function
    df = get_text()
    print(df.head())


