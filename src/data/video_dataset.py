from pathlib import Path

import torch
from torch.utils.data import Dataset


LABEL_MAP = {
    "Normal": 0,
    "Fighting": 1,
    "Shooting": 2,
    "Explosion": 3,
}


class VideoFeatureDataset(Dataset):

    def __init__(self, csv_file, feature_dir):

        self.data = __import__("pandas").read_csv(csv_file)
        self.feature_dir = Path(feature_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        video_id = row["video_id"]
        label_name = row["label"]

        feature_file = (
            self.feature_dir / f"{video_id}.pt"
        )

        data = torch.load(
            feature_file,
            weights_only=False,
        )

        features = data["features"]

        label = LABEL_MAP[label_name]

        return features, torch.tensor(label, dtype=torch.long)
    