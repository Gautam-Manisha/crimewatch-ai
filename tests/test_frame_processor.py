import pandas as pd
import torch

from src.preprocessing.sampler import sample_uniform_frames
from src.preprocessing.frame_processor import preprocess_frames


def test_frame_preprocessing():

    df = pd.read_csv("data/metadata/train.csv")

    video_path = df.iloc[0]["file_path"]

    frames, _ = sample_uniform_frames(
        video_path,
        num_samples=20,
    )

    tensor = preprocess_frames(frames)

    print("\nTensor shape:", tensor.shape)
    print("Tensor dtype:", tensor.dtype)
    print("Min value:", tensor.min().item())
    print("Max value:", tensor.max().item())

    assert tensor.shape == (20, 3, 224, 224)
    assert tensor.dtype == torch.float32

    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0