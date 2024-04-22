import os
import wget
import argparse


def download_squad_dataset(path: str):
    assert os.path.exists(path), f"The specified path does not exist: {path}."
    files = ['train-v1.1.json', 'dev-v1.1.json', 'evaluate-v1.1.py']
    for filename, url_link in files:
        wget.download(url_link, os.path.join(path, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The folder to download the train, validation and test datasets.")
    args = parser.parse_args()
    download_squad_dataset(args.path)
