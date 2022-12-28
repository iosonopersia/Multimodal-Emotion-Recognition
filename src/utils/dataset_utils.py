import os
import pandas as pd


def get_text (mode="train"):
    """Returns the transcripts for the given mode (train, val, test)."""
    assert mode in ["train", "val", "test"]
    root = os.path.join(os.path.abspath("data"), "MELD.Raw")

    if mode == "train":
        data_path = os.path.join(root, "train_sent_emo.csv")
    elif mode == "val":
        data_path = os.path.join(root, "dev_sent_emo.csv")
    elif mode == 'test':
        data_path = os.path.join(root, "test_sent_emo.csv")

    if not os.path.exists(data_path):
        raise ValueError("Dataset not found at {}".format(data_path))

    df = pd.read_csv(data_path, usecols=['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID'])

    cp1252_to_utf8 = {
        "\x85": "…", # \x85 \u2026 HORIZONTAL ELLIPSIS
        "\x91": "‘", # \x91 \u2018 LEFT SINGLE QUOTATION MARK
        "\x92": "’", # \x92 \u2019 RIGHT SINGLE QUOTATION MARK
        "\x93": "“", # \x93 \u201C LEFT DOUBLE QUOTATION MARK
        "\x94": "”", # \x94 \u201D RIGHT DOUBLE QUOTATION MARK
        "\x96": "–", # \x96 \u2013 EN DASH
        "\x97": "—", # \x97 \u2014 EM DASH
        "\xa0": " ", # \xa0 \u00A0 NO-BREAK SPACE
    }
    for key, value in cp1252_to_utf8.items():
        df["Utterance"] = df["Utterance"].map(lambda x: x.replace(key, value))

    return df


if __name__ == "__main__":

    # Test get_text function
    df = get_text()
    print(df.head())


