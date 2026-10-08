import torch

from src.models.cnn_lstm import CNNLSTM


def test_cnn_lstm():

    model = CNNLSTM()

    x = torch.randn(
        8,
        20,
        512,
    )

    output = model(x)

    print("\nInput shape:", x.shape)
    print("Output shape:", output.shape)

    assert output.shape == (8, 4)


if __name__ == "__main__":
    test_cnn_lstm()