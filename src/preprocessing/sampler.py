import cv2
import numpy as np


def sample_uniform_frames(
    video_path: str,
    num_samples: int = 20,
):
    """
    Uniformly sample frames from a video.

    Returns:
        frames: list of sampled BGR frames
        timestamps: corresponding timestamps in seconds
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        raise ValueError(
            f"Invalid frame count for: {video_path}"
        )

    if fps <= 0:
        cap.release()
        raise ValueError(
            f"Invalid FPS for: {video_path}"
        )

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_samples,
        dtype=int,
    )

    frames = []
    timestamps = []

    for frame_index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        success, frame = cap.read()

        if not success:
            continue

        timestamp = frame_index / fps

        frames.append(frame)
        timestamps.append(timestamp)

    cap.release()

    return frames, timestamps