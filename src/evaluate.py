import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.video_dataset import VideoFeatureDataset
from src.models.cnn_lstm import CNNLSTM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/cnn_lstm_best.pth"

TEST_CSV = "data/metadata/test.csv"

FEATURE_DIR = "data/processed/cnn_features/test"

OUTPUT_DIR = Path("outputs/evaluation")

LABELS = [
    "Normal",
    "Fighting",
    "Shooting",
    "Explosion",
]

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# EVALUATOR
# ============================================================

class CrimeWatchEvaluator:

    def __init__(self):

        print("=" * 70)
        print("CRIMEWATCH AI - MODEL EVALUATION")
        print("=" * 70)

        print("Device:", DEVICE)

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        self.dataset = VideoFeatureDataset(
            TEST_CSV,
            FEATURE_DIR
        )

        print(
            "Test samples:",
            len(self.dataset)
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self.model = CNNLSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            num_classes=4,
            dropout=0.3,
        )

        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(DEVICE)

        self.model.eval()

        print("Model loaded successfully.")


    # ========================================================
    # RUN EVALUATION
    # ========================================================

    def evaluate(self):

        true_labels = []
        predicted_labels = []

        print()
        print(
            "Running inference on test set..."
        )

        with torch.no_grad():

            for index in range(
                len(self.dataset)
            ):

                features, label = self.dataset[
                    index
                ]

                # Add batch dimension
                features = features.unsqueeze(0)

                features = features.to(
                    DEVICE
                )

                output = self.model(
                    features
                )

                prediction = torch.argmax(
                    output,
                    dim=1
                ).item()

                true_label = label.item()

                true_labels.append(
                    true_label
                )

                predicted_labels.append(
                    prediction
                )

        return (
            true_labels,
            predicted_labels
        )


    # ========================================================
    # METRICS
    # ========================================================

    def calculate_metrics(
        self,
        true_labels,
        predicted_labels
    ):

        accuracy = accuracy_score(
            true_labels,
            predicted_labels
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                true_labels,
                predicted_labels,
                labels=[0, 1, 2, 3],
                average="weighted",
                zero_division=0,
            )
        )

        report = classification_report(
            true_labels,
            predicted_labels,
            labels=[0, 1, 2, 3],
            target_names=LABELS,
            zero_division=0,
        )

        matrix = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=[0, 1, 2, 3],
        )

        return {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "classification_report": report,
            "confusion_matrix": matrix,
        }


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    def save_confusion_matrix(
        self,
        matrix
    ):

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        image = ax.imshow(
            matrix
        )

        ax.set_title(
            "CrimeWatch AI - Confusion Matrix"
        )

        ax.set_xlabel(
            "Predicted Label"
        )

        ax.set_ylabel(
            "True Label"
        )

        ax.set_xticks(
            range(len(LABELS))
        )

        ax.set_yticks(
            range(len(LABELS))
        )

        ax.set_xticklabels(
            LABELS,
            rotation=30,
            ha="right"
        )

        ax.set_yticklabels(
            LABELS
        )

        # Add numbers to cells
        for i in range(
            len(LABELS)
        ):

            for j in range(
                len(LABELS)
            ):

                ax.text(
                    j,
                    i,
                    matrix[i, j],
                    ha="center",
                    va="center"
                )

        fig.colorbar(
            image,
            ax=ax
        )

        fig.tight_layout()

        output_path = (
            OUTPUT_DIR
            / "confusion_matrix.png"
        )

        fig.savefig(
            output_path,
            dpi=200
        )

        plt.close(fig)

        return output_path


    # ========================================================
    # SAVE METRICS
    # ========================================================

    def save_metrics(
        self,
        metrics
    ):

        output = {
            "test_accuracy": float(
                metrics["accuracy"]
            ),
            "weighted_precision": float(
                metrics[
                    "precision_weighted"
                ]
            ),
            "weighted_recall": float(
                metrics[
                    "recall_weighted"
                ]
            ),
            "weighted_f1": float(
                metrics[
                    "f1_weighted"
                ]
            ),
            "confusion_matrix":
                metrics[
                    "confusion_matrix"
                ].tolist(),
            "classification_report":
                metrics[
                    "classification_report"
                ],
        }

        output_path = (
            OUTPUT_DIR
            / "metrics.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output,
                file,
                indent=4
            )

        return output_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluator = CrimeWatchEvaluator()

    true_labels, predicted_labels = (
        evaluator.evaluate()
    )

    metrics = evaluator.calculate_metrics(
        true_labels,
        predicted_labels
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{metrics['precision_weighted'] * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{metrics['recall_weighted'] * 100:.2f}%"
    )

    print(
        f"F1 Score : "
        f"{metrics['f1_weighted'] * 100:.2f}%"
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        metrics["classification_report"]
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        metrics["confusion_matrix"]
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    matrix_path = (
        evaluator.save_confusion_matrix(
            metrics["confusion_matrix"]
        )
    )

    metrics_path = (
        evaluator.save_metrics(
            metrics
        )
    )

    print()
    print(
        "Confusion matrix saved:",
        matrix_path
    )

    print(
        "Metrics saved:",
        metrics_path
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)