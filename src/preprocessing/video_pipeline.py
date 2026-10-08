from src.preprocessing.sampler import sample_uniform_frames
from src.preprocessing.frame_processor import preprocess_frames


def load_video_tensor(
    video_path: str,
    num_samples: int = 20,
):
    """
    Complete video preprocessing pipeline.

    Video
        ↓
    Uniform frame sampling
        ↓
    Frame preprocessing
        ↓
    PyTorch tensor

    Returns:
        Tensor of shape:
        (num_samples, 3, 224, 224)
    """

    frames, timestamps = sample_uniform_frames(
        video_path,
        num_samples=num_samples,
    )

    tensor = preprocess_frames(frames)

    return tensor, timestamps