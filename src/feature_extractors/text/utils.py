from munch import Munch
import os
import pandas as pd


config = None

def get_config():
    global config
    if config is None:
        with open('./src/feature_extractors/text/config.yaml', 'rt', encoding='utf-8') as f:
            config = Munch.fromYAML(f.read())
    return config


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

    # Remove corrupted multimedia files
    if mode == "train":
        df = df[(df["Dialogue_ID"] != 125) | (df["Utterance_ID"] != 3)]
    elif mode == "val":
        df = df[(df["Dialogue_ID"] != 110) | (df["Utterance_ID"] != 7)]
    elif mode == "test":
        df = df[(df["Dialogue_ID"] != 38) | (df["Utterance_ID"] != 4)]
        df = df[(df["Dialogue_ID"] != 220) | (df["Utterance_ID"] != 0)]
    df = df.reset_index(drop=True)

    # Convert encoding from cp1252 to utf-8
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

def get_utterance_with_context(df, idx, separator):
    main_utterance_row = df.iloc[idx]
    dialogue_id = int(main_utterance_row["Dialogue_ID"])
    main_utterance_id = int(main_utterance_row["Utterance_ID"])

    dialogue = df[df["Dialogue_ID"] == dialogue_id]
    dia_utt_ids = sorted(dialogue["Utterance_ID"].to_list())
    try:
        main_utt_idx_in_dialogue = dia_utt_ids.index(main_utterance_id)
    except ValueError:
        raise ValueError(f"Utterance ID {main_utterance_id} not found in dialogue ID {dialogue_id}")
    prev_utterance_id = dia_utt_ids[main_utt_idx_in_dialogue - 1] if main_utt_idx_in_dialogue > 0 else None
    next_utterance_id = dia_utt_ids[main_utt_idx_in_dialogue + 1] if main_utt_idx_in_dialogue < len(dia_utt_ids) - 1 else None

    # Concatenate the previous and next utterances
    utterance_with_context = main_utterance_row["Utterance"]

    if prev_utterance_id is not None:
        prev_utterance_row = dialogue[dialogue["Utterance_ID"] == prev_utterance_id].iloc[0]
        prev_utterance = prev_utterance_row["Utterance"]
        utterance_with_context = f"{prev_utterance} {separator} {utterance_with_context}"
    else:
        utterance_with_context = f"{separator} {utterance_with_context}"

    if next_utterance_id is not None:
        next_utterance_row = dialogue[dialogue["Utterance_ID"] == next_utterance_id].iloc[0]
        next_utterance = next_utterance_row["Utterance"]
        utterance_with_context = f"{utterance_with_context} {separator} {next_utterance}"
    else:
        utterance_with_context = f"{utterance_with_context} {separator}"

    return utterance_with_context
