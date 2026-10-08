import torch
import torch.nn as nn


def test_mps_tensor_operation():

    if not torch.backends.mps.is_available():
        print("MPS is not available; test skipped.")
        return

    device = torch.device("mps")

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    ).to(device)

    x = torch.randn(4, 10).to(device)

    output = model(x)

    print("\n=== MPS Neural Network Test ===")
    print("Device:", output.device)
    print("Output shape:", output.shape)

    assert output.shape == (4, 2)