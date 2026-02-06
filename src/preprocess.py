import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw/catsvsdogs/train"
PROCESSED_DIR = "data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")

SPLIT_RATIO = 0.2

def prepare_dirs():
    for path in [
        TRAIN_DIR + "/cats",
        TRAIN_DIR + "/dogs",
        VAL_DIR + "/cats",
        VAL_DIR + "/dogs",
    ]:
        os.makedirs(path, exist_ok=True)

def split_and_copy(class_name):
    files = os.listdir(os.path.join(RAW_DIR, class_name))
    train_files, val_files = train_test_split(
        files, test_size=SPLIT_RATIO, random_state=42
    )

    for f in train_files:
        shutil.copy(
            os.path.join(RAW_DIR, class_name, f),
            os.path.join(TRAIN_DIR, class_name, f),
        )

    for f in val_files:
        shutil.copy(
            os.path.join(RAW_DIR, class_name, f),
            os.path.join(VAL_DIR, class_name, f),
        )

def main():
    prepare_dirs()
    split_and_copy("cats")
    split_and_copy("dogs")
    print("Preprocessing completed")

if __name__ == "__main__":
    main()
