import os
from tqdm import tqdm
import argparse
import subprocess
import logging


def convert_videos(mp4_path):
    mp4_path = os.path.abspath(mp4_path) # Get absolute path

    # Create wav folder if it doesn't exist
    wav_path = os.path.join(mp4_path, "wav")
    if not os.path.exists(wav_path):
        os.makedirs(wav_path)

    already_converted = set(filter(lambda x: x.endswith(".wav"), os.listdir(wav_path)))
    already_converted = set(map(lambda x: x.replace(".wav", ".mp4"), already_converted))

    mp4s = os.listdir(mp4_path)
    mp4s = set(filter(lambda x: not x.startswith(".") and x.endswith(".mp4"), mp4s))
    mp4s.difference_update(already_converted)

    print(f"Found {len(mp4s)} mp4 file", end="")
    print("s"if len(mp4s)>1 else "", end="")
    print(f" in {mp4_path} still to be converted...")

    for mp4 in tqdm(mp4s):
        wav_file = mp4.replace(".mp4", ".wav")
        input_file = os.path.join(mp4_path, mp4)
        output_file = os.path.join(wav_path, wav_file)

        command = f"ffmpeg -y -f mp4 -i \"{input_file}\" -ac 1 -ar 16000 -vn -f wav \"{output_file}\" > .stdout 2> .stderr"
        retcode = subprocess.call(command, shell=True)

        if retcode != 0:
            with open(".stderr", "r") as f:
                logging.error(f"[{mp4}] {f.read()}")

        os.remove(".stdout")
        os.remove(".stderr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--train_path", type=str, default="data/MELD.Raw/train_splits")
    parser.add_argument("-b", "--dev_path", type=str, default="data/MELD.Raw/dev_splits_complete")
    parser.add_argument("-c", "--test_path", type=str, default="data/MELD.Raw/output_repeated_splits_test")
    args = parser.parse_args()

    logging.basicConfig(filename="mp4towav_log.txt", filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

    convert_videos(args.train_path)
    convert_videos(args.dev_path)
    convert_videos(args.test_path)
