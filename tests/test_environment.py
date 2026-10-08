import cv2
import numpy as np
import torch


def test_environment():
    print("\n=== CrimeWatch AI Environment ===")

    print(f"PyTorch version : {torch.__version__}")
    print(f"OpenCV version  : {cv2.__version__}")
    print(f"NumPy version   : {np.__version__}")

    print(f"MPS built       : {torch.backends.mps.is_built()}")
    print(f"MPS available   : {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        tensor = torch.randn(3, 3, device=device)

        print(f"Device          : {device}")
        print(f"Tensor shape    : {tensor.shape}")

    else:
        device = torch.device("cpu")
        tensor = torch.randn(3, 3)

        print("MPS unavailable. Falling back to CPU.")

    assert tensor.shape == (3, 3)