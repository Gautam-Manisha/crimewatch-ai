import cv2
from pathlib import Path


OUTPUT_DIR = Path("outputs/evidence")


def extract_evidence_frame(
    video_path,
    timestamp,
    incident_id
):
    """
    Extract a representative frame from a video
    at the given timestamp.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    # Move video to requested timestamp
    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        timestamp * 1000
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            f"Could not extract frame at "
            f"{timestamp:.2f}s"
        )

    output_path = (
        OUTPUT_DIR /
        f"{incident_id}_frame.jpg"
    )

    cv2.imwrite(
        str(output_path),
        frame
    )

    return output_path


def extract_video_segment(
    video_path,
    start_time,
    end_time,
    incident_id
):
    """
    Extract the video segment corresponding
    to a detected incident.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0:
        raise RuntimeError(
            "Could not determine video FPS"
        )

    output_path = (
        OUTPUT_DIR /
        f"{incident_id}_segment.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    # Start at incident beginning
    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        start_time * 1000
    )

    while True:

        current_time = (
            cap.get(
                cv2.CAP_PROP_POS_MSEC
            ) / 1000
        )

        if current_time > end_time:
            break

        success, frame = cap.read()

        if not success:
            break

        writer.write(frame)

    cap.release()
    writer.release()

    return output_path