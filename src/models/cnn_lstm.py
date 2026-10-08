import torch
import torch.nn as nn


class CNNLSTM(nn.Module):

    def __init__(
        self,
        input_size=512,
        hidden_size=256,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        """
        Input:
            x: (batch, sequence_length, input_size)

        Example:
            (8, 20, 512)

        Output:
            logits: (batch, num_classes)
        """

        lstm_output, _ = self.lstm(x)

        # Take the final time-step representation
        final_output = lstm_output[:, -1, :]

        logits = self.classifier(final_output)

        return logits