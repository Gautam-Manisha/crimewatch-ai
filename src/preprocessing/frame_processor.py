import cv2
import numpy as np
import torch


IMAGE_SIZE = 224


def preprocess_frames(frames):
    """
    Preprocess sampled video frames.

    Input:
        frames: list of OpenCV BGR frames

    Returns:
        Tensor of shape:
        (num_frames, 3, 224, 224)
    """

    processed = []

    for frame in frames:

        # OpenCV: BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        # Convert uint8 [0,255] -> float32 [0,1]
        frame = frame.astype(np.float32) / 255.0

        # HWC -> CHW
        frame = np.transpose(frame, (2, 0, 1))

        processed.append(frame)

    frames_array = np.stack(processed)

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(frames_array)

    return tensor