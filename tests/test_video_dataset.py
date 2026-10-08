import torch
from torch.utils.data import DataLoader

from src.data.video_dataset import VideoFeatureDataset


def test_video_dataset():

    dataset = VideoFeatureDataset(
        csv_file="data/metadata/train.csv",
        feature_dir="data/processed/cnn_features/train",
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    features, labels = next(iter(loader))

    print("\nDataset size:", len(dataset))
    print("Feature batch:", features.shape)
    print("Label batch:", labels.shape)
    print("Labels:", labels)

    assert len(dataset) == 120

    assert features.shape == (
        8,
        20,
        512,
    )

    assert labels.shape == (8,)

    assert features.dtype == torch.float32
    assert labels.dtype == torch.long