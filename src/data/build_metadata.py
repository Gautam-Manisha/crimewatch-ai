from pathlib import Path
import cv2
import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def get_video_metadata(video_path: Path) -> dict:
    """
    Extract technical metadata from a single video.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0.0

    cap.release()

    return {
        "fps": fps,
        "num_frames": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
    }


def infer_label(video_path: Path) -> str:
    """
    Infer the activity label from the parent directory.

    Example:
        data/raw/UCF_Crime/Fighting/video01.mp4

    returns:
        Fighting
    """

    return video_path.parent.name


def build_metadata(data_dir: str, output_file: str) -> None:

    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    records = []

    video_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    print(f"Found {len(video_files)} video files.")

    for index, video_path in enumerate(video_files, start=1):

        print(
            f"[{index}/{len(video_files)}] "
            f"Processing {video_path.name}"
        )

        try:
            metadata = get_video_metadata(video_path)

            record = {
                "video_id": video_path.stem,
                "file_path": str(video_path),
                "label": infer_label(video_path),
                **metadata,
            }

            records.append(record)

        except Exception as exc:
            print(
                f"WARNING: Could not process "
                f"{video_path}: {exc}"
            )

    df = pd.DataFrame(records)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print()
    print("===================================")
    print("Metadata generation complete")
    print("===================================")
    print(f"Videos processed : {len(df)}")
    print(f"Output file      : {output_path}")


if __name__ == "__main__":

    build_metadata(
        data_dir="data/raw/UCF_Crime",
        output_file="data/metadata/videos.csv",
    )