import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Dataset paths (match your structure)
RAW_DIR = Path("data/raw/catsvsdogs/train")
PROCESSED_DIR = Path("data/processed")

CLASSES = ["cats", "dogs"]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def check_raw_data():
    """
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
    Create processed train/val directories.
    """
    for split in ["train", "val"]:
        for cls in CLASSES:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def split_and_copy(class_name):
    """
    Split images into train and validation sets and copy them.
    """
    src_dir = RAW_DIR / class_name
    images = list(src_dir.glob("*.jpg"))

    train_imgs, val_imgs = train_test_split(
        images,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    for img in train_imgs:
        shutil.copy(img, PROCESSED_DIR / "train" / class_name / img.name)

    for img in val_imgs:
        shutil.copy(img, PROCESSED_DIR / "val" / class_name / img.name)


def main():
    print("Checking raw data...")
    check_raw_data()

    print("Preparing directories...")
    prepare_dirs()

    print("Splitting and copying images...")
    for cls in CLASSES:
        split_and_copy(cls)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
