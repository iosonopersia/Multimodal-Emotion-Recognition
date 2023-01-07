from munch import Munch
import os
import pandas as pd


config = None

def get_config():
    global config
    if config is None:
        with open('./src/feature_extractors/audio_wav2vec2/config.yaml', 'rt', encoding='utf-8') as f:
            config = Munch.fromYAML(f.read())
    return config


def get_text(mode="train"):
    """Returns the transcripts for the given mode (train, val, test)."""
    assert mode in ["train", "val", "test"]
    root = os.path.join(os.path.abspath("data"), "MELD.Raw")

    if mode == "train":
        data_path = os.path.join(root, "train_sent_emo.csv")
    elif mode == "val":
        data_path = os.path.join(root, "dev_sent_emo.csv")
    elif mode == 'test':
        data_path = os.path.join(root, "test_sent_emo.csv")
    else:
        raise ValueError(f"Invalid mode {mode}")

    if not os.path.exists(data_path):
        raise ValueError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path, usecols=['Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID'])

    # Remove corrupted multimedia files
    if mode == "train":
        df = df[(df["Dialogue_ID"] != 125) | (df["Utterance_ID"] != 3)]
    elif mode == "val":
        df = df[(df["Dialogue_ID"] != 110) | (df["Utterance_ID"] != 7)]
    elif mode == "test":
        df = df[(df["Dialogue_ID"] != 38) | (df["Utterance_ID"] != 4)]
        df = df[(df["Dialogue_ID"] != 220) | (df["Utterance_ID"] != 0)]
    df = df.reset_index(drop=True)

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
