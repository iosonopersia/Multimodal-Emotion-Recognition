import os
from tqdm import tqdm
import argparse
import subprocess
import logging


def convert_videos(mp4_path):
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

    mp4s = {os.path.join(mp4_path, mp4) for mp4 in mp4s}
    for mp4 in tqdm(mp4s):
        wav_filename = os.path.basename(mp4).replace("mp4", "wav")
        output_file = os.path.join(wav_path, wav_filename)

        with open(".stdout", "w") as f, open(".stderr", "w") as g:
            retcode = subprocess.call([args.ffmpeg_path, "-y", "-f", "mp4", "-i", mp4, "-codec", "copy", "-f", "wav", output_file], stdout=f, stderr=g)

        if retcode != 0:
            with open(".stderr", "r") as f:
                logging.error(f"[{mp4}] {f.read()}")

        os.remove(".stdout")
        os.remove(".stderr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--ffmpeg_path", type=str, default="C://ffmpeg/bin/ffmpeg.exe")
    parser.add_argument("-a", "--train_path", type=str, default="C://MELD/train_splits")
    parser.add_argument("-b", "--dev_path", type=str, default="C://MELD/dev_splits_complete")
    parser.add_argument("-c", "--test_path", type=str, default="C://MELD/output_repeated_splits_test")
    args = parser.parse_args()

    logging.basicConfig(filename="mp4towav_log.txt", filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

    convert_videos(args.train_path)
    convert_videos(args.dev_path)
    convert_videos(args.test_path)
