from pathlib import Path

import pandas as pd
import torch

from src.models.cnn_feature_extractor import CNNFeatureExtractor
from src.features.extract_cnn_features import extract_video_features


INPUT_FILES = [
    "data/metadata/train.csv",
    "data/metadata/val.csv",
    "data/metadata/test.csv",
]

OUTPUT_DIR = Path("data/processed/cnn_features")

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def build_features():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load CNN
    model = CNNFeatureExtractor()
    model = model.to(DEVICE)
    model.eval()

    total = 0

    for input_file in INPUT_FILES:

        df = pd.read_csv(input_file)

        split_name = Path(input_file).stem

        print()
        print("=" * 50)
        print(f"Processing {split_name}")
        print("=" * 50)

        split_output = OUTPUT_DIR / split_name
        split_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, row in df.iterrows():

            video_path = row["file_path"]
            video_id = row["video_id"]

            output_file = (
                split_output / f"{video_id}.pt"
            )

            # Skip if already processed
            if output_file.exists():
                print(
                    f"[{index + 1}/{len(df)}] "
                    f"Already exists: {video_id}"
                )
                continue

            print(
                f"[{index + 1}/{len(df)}] "
                f"Processing: {video_id}"
            )

            try:

                features, timestamps = (
                    extract_video_features(
                        video_path,
                        model,
                        num_samples=20,
                    )
                )

                torch.save(
                    {
                        "features": features,
                        "timestamps": timestamps,
                        "label": row["label"],
                        "video_id": video_id,
                        "video_path": video_path,
                    },
                    output_file,
                )

                total += 1

            except Exception as exc:

                print(
                    f"ERROR processing "
                    f"{video_id}: {exc}"
                )

    print()
    print("=" * 50)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 50)
    print("New videos processed:", total)


if __name__ == "__main__":
    build_features()