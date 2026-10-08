from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/metadata/videos.csv"

TRAIN_FILE = "data/metadata/train.csv"
VAL_FILE = "data/metadata/val.csv"
TEST_FILE = "data/metadata/test.csv"

RANDOM_STATE = 42


def create_splits():

    df = pd.read_csv(INPUT_FILE)

    print("Total videos:", len(df))
    print("\nOriginal class distribution:")
    print(df["label"].value_counts())

    # First: 80% development pool, 20% test
    train_val, test = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # Then: 10% validation overall
    # 75% of the remaining 80% = 60% overall train
    # 25% of the remaining 80% = 20% overall validation
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_val["label"],
    )

    train.to_csv(TRAIN_FILE, index=False)
    val.to_csv(VAL_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    print("\n==============================")
    print("Split created")
    print("==============================")

    print("\nTrain:")
    print(train["label"].value_counts())

    print("\nValidation:")
    print(val["label"].value_counts())

    print("\nTest:")
    print(test["label"].value_counts())

    print("\nTotal:")
    print(
        f"Train: {len(train)} | "
        f"Validation: {len(val)} | "
        f"Test: {len(test)}"
    )


if __name__ == "__main__":
    create_splits()