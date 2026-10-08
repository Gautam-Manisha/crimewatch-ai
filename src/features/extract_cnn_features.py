from pathlib import Path

import torch

from src.models.cnn_feature_extractor import CNNFeatureExtractor
from src.preprocessing.video_pipeline import load_video_tensor


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def extract_video_features(
    video_path: str,
    model: CNNFeatureExtractor,
    num_samples: int = 20,
):
    """
    Convert one video into a sequence of CNN features.

    Input:
        video_path -> path to video

    Output:
        features -> (20, 512)
    """

    # Video -> preprocessed tensor
    frames, timestamps = load_video_tensor(
        video_path,
        num_samples=num_samples,
    )

    # Move frames to device
    frames = frames.to(DEVICE)

    # CNN inference
    with torch.no_grad():
        features = model(frames)

    # Move back to CPU
    features = features.cpu()

    return features, timestamps


if __name__ == "__main__":

    import pandas as pd

    df = pd.read_csv(
        "data/metadata/train.csv"
    )

    video_path = df.iloc[0]["file_path"]

    print("Device:", DEVICE)
    print("Video:", video_path)

    model = CNNFeatureExtractor()
    model = model.to(DEVICE)
    model.eval()

    features, timestamps = extract_video_features(
        video_path,
        model,
        num_samples=20,
    )

    print()
    print("Feature shape:", features.shape)
    print("Timestamps:", len(timestamps))