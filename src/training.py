from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.video_dataset import VideoFeatureDataset
from src.models.cnn_lstm import CNNLSTM


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 1e-4

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def create_dataloaders():

    train_dataset = VideoFeatureDataset(
        csv_file="data/metadata/train.csv",
        feature_dir="data/processed/cnn_features/train",
    )

    val_dataset = VideoFeatureDataset(
        csv_file="data/metadata/val.csv",
        feature_dir="data/processed/cnn_features/val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader


def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = correct / total

    return average_loss, accuracy


def train():

    print("Device:", DEVICE)

    train_loader, val_loader = (
        create_dataloaders()
    )

    model = CNNLSTM()

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = (
            running_loss / len(train_loader)
        )

        train_accuracy = (
            correct / total
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                MODEL_DIR / "cnn_lstm_best.pth",
            )

            print(
                "  ✓ Best model saved"
            )

    print()
    print("==============================")
    print("TRAINING COMPLETE")
    print("==============================")
    print(
        "Best validation accuracy:",
        best_val_accuracy,
    )


if __name__ == "__main__":
    train()