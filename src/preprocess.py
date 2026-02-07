import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
RAW_DIR = Path("data/raw/catsvsdogs/train")
PROCESSED_DIR = Path("data/processed")

CLASSES = ["cats", "dogs"]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def check_raw_data():
    """
    Ensure raw data exists.
    Raw data is treated as an external dependency (Kaggle).
    """
    if not RAW_DIR.exists():
        raise RuntimeError(
            "Raw data not found.\n"
            "Please download the dataset from Kaggle:\n"
            "kaggle datasets download -d salader/dogs-vs-cats -p data/raw\n"
            "unzip data/raw/dogs-vs-cats.zip -d data/raw"
        )


def prepare_dirs():
    """
    Create processed data directories.
    """
    for split in ["train", "val"]:
        for cls in CLASSES:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def split_and_copy(class_name):
    """
    Split images i
