
from pathlib import Path
from collections import deque
import time

import cv2
import numpy as np
import streamlit as st
import torch

from src.models.cnn_feature_extractor import CNNFeatureExtractor
from src.models.cnn_lstm import CNNLSTM
from src.preprocessing.frame_processor import preprocess_frames


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATA_DIR = Path("data/raw/UCF_Crime")
MODEL_PATH = Path("models/cnn_lstm_best.pth")

CLASSES = ["Normal", "Fighting", "Shooting", "Explosion"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

NUM_FRAMES = 20
PREDICT_EVERY = 5

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

st.set_page_config(
    page_title="CrimeWatch AI | Live Detection",
    page_icon="🎥",
    layout="wide",
)


# ---------------------------------------------------------
# LOAD MODEL ONCE
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    cnn = CNNFeatureExtractor().to(DEVICE)
    cnn.eval()

    lstm = CNNLSTM(
        input_size=512,
        hidden_size=256,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
    )

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    lstm.load_state_dict(state)
    lstm.to(DEVICE)
    lstm.eval()

    return cnn, lstm


def get_videos(class_name):
    folder = DATA_DIR / class_name

    if not folder.exists():
        return []

    return sorted(
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def predict_window(frames, cnn, lstm):
    """
    Run the existing CNN-LSTM on one temporal window.
    Frames must be OpenCV BGR images.
    """

    processed = preprocess_frames(frames)

    if not isinstance(processed, torch.Tensor):
        processed = torch.as_tensor(processed)

    processed = processed.to(
        device=DEVICE,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        features = cnn(processed)

        # Shape: (1, sequence_length, 512)
        features = features.unsqueeze(0)

        logits = lstm(features)
        probabilities = torch.softmax(logits, dim=1)[0]

        confidence, class_index = probabilities.max(dim=0)

    return (
        CLASSES[class_index.item()],
        float(confidence.item()) * 100,
        probabilities.detach().cpu().numpy(),
    )


def overlay_prediction(frame, label, confidence, timestamp, device):
    output = frame.copy()

    color = (
        (0, 210, 0)
        if label == "Normal"
        else (0, 0, 255)
    )

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 105),
        (20, 20, 20),
        thickness=-1,
    )

    cv2.putText(
        output,
        f"ACTIVITY: {label.upper()}",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        f"Confidence: {confidence:.1f}%  |  Time: {timestamp:.1f}s",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        f"Device: {device}",
        (15, 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------

st.title("CrimeWatch AI — Live Video Detection")

st.write(
    "Select a video from any class folder. "
    "Predictions update as consecutive video frames are processed."
)

st.caption(
    "This is live inference over a stored video, not a live CCTV/IP camera feed."
)

st.write(f"Compute device: **{DEVICE}**")

left, right = st.columns(2)

with left:
    selected_class = st.selectbox(
        "Choose dataset folder",
        CLASSES,
    )

with right:
    videos = get_videos(selected_class)

    if videos:
        selected_video = st.selectbox(
            "Choose video",
            videos,
            format_func=lambda path: path.name,
        )
    else:
        selected_video = None
        st.error(f"No supported videos found in {DATA_DIR / selected_class}")


if selected_video:
    st.info(f"Selected video: `{selected_video.name}`")

    if st.button("▶ Start Live Detection", type="primary"):
        try:
            cnn, lstm = load_models()

            cap = cv2.VideoCapture(str(selected_video))

            if not cap.isOpened():
                st.error("Could not open the selected video.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                fps = 25.0

            video_display = st.empty()

            metric_cols = st.columns(3)

            predicted_class_metric = metric_cols[0].empty()
            confidence_metric = metric_cols[1].empty()
            timestamp_metric = metric_cols[2].empty()

            confidence_bar = st.progress(0)
            probability_area = st.empty()
            status = st.empty()
            progress = st.progress(0)

            frame_buffer = deque(maxlen=NUM_FRAMES)

            latest_label = "Collecting frames..."
            latest_confidence = 0.0
            latest_probabilities = None
            frame_index = 0

            status.info(
                f"Loading temporal window: 0/{NUM_FRAMES} frames"
            )

            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame_index += 1
                timestamp = frame_index / fps

                frame_buffer.append(frame)

                # Wait until a full temporal sequence exists.
                # Then update the prediction every PREDICT_EVERY frames.
                should_predict = (
                    len(frame_buffer) == NUM_FRAMES
                    and (
                        latest_probabilities is None
                        or frame_index % PREDICT_EVERY == 0
                    )
                )

                if should_predict:
                    (
                        latest_label,
                        latest_confidence,
                        latest_probabilities,
                    ) = predict_window(
                        list(frame_buffer),
                        cnn,
                        lstm,
                    )

                    predicted_class_metric.metric(
                        "Predicted activity",
                        latest_label,
                    )

                    confidence_metric.metric(
                        "Model confidence",
                        f"{latest_confidence:.1f}%",
                    )

                    timestamp_metric.metric(
                        "Video timestamp",
                        f"{timestamp:.1f}s",
                    )

                    confidence_bar.progress(
                        min(
                            100,
                            max(0, int(latest_confidence)),
                        )
                    )

                    probability_area.write("Class probabilities")

                    probability_area.bar_chart(
                        {
                            "Probability (%)": {
                                class_name: float(
                                    latest_probabilities[i] * 100
                                )
                                for i, class_name in enumerate(CLASSES)
                            }
                        }
                    )

                    status.success(
                        f"Prediction updated at {timestamp:.1f}s"
                    )

                if len(frame_buffer) < NUM_FRAMES:
                    display_label = "Collecting frames"
                    display_confidence = 0.0
                else:
                    display_label = latest_label
                    display_confidence = latest_confidence

                annotated = overlay_prediction(
                    frame,
                    display_label,
                    display_confidence,
                    timestamp,
                    DEVICE.type,
                )

                # Resize only for display to reduce UI overhead.
                display_frame = cv2.resize(
                    annotated,
                    (960, int(annotated.shape[0] * 960 / annotated.shape[1])),
                )

                video_display.image(
                    display_frame,
                    channels="RGB",
                    use_container_width=True,
                )

                if total_frames > 0:
                    progress.progress(
                        min(1.0, frame_index / total_frames)
                    )

            cap.release()

            status.success(
                "Video finished. Select another class/video to test again."
            )

        except Exception as error:
            st.error(f"Live inference failed: {error}")
            st.exception(error)

