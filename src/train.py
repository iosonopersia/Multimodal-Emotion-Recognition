import argparse
from utils import get_config


def main(args):
    config = get_config()
    print('Hello, world!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
