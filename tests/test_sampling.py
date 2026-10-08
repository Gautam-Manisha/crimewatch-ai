import pandas as pd

from src.preprocessing.sampler import sample_uniform_frames


def test_uniform_sampling():

    df = pd.read_csv("data/metadata/train.csv")

    video_path = df.iloc[0]["file_path"]

    frames, timestamps = sample_uniform_frames(
        video_path,
        num_samples=20,
    )

    print("\nVideo:", video_path)
    print("Frames sampled:", len(frames))
    print("Timestamps:", timestamps)

    assert len(frames) == 20
    assert len(timestamps) == 20