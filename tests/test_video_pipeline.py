import pandas as pd
import torch

from src.preprocessing.video_pipeline import load_video_tensor


def test_video_pipeline():

    df = pd.read_csv("data/metadata/train.csv")

    video_path = df.iloc[0]["file_path"]

    tensor, timestamps = load_video_tensor(
        video_path,
        num_samples=20,
    )

    print("\nVideo:", video_path)
    print("Tensor shape:", tensor.shape)
    print("Timestamps:", timestamps)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (20, 3, 224, 224)
    assert len(timestamps) == 20

    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0