
from collections import deque
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import torch

from src.analysis.video_analyzer import CrimeVideoAnalyzer


CLASSES = ["Normal", "Fighting", "Shooting", "Explosion"]
NUM_FRAMES = 20
INFERENCE_EVERY = 5


def _overlay(frame, label, confidence, timestamp, status):
    frame = frame.copy()
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 94), (20, 20, 20), -1)

    color = (60, 220, 100) if label == "Normal" else (40, 70, 255)

    cv2.putText(
        frame, f"ACTIVITY: {label}", (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"Confidence: {confidence:.1f}%  |  {timestamp:.1f}s",
        (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, status, (12, 82), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (220, 220, 220), 1, cv2.LINE_AA
    )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_live_monitor():
    st.subheader("CrimeWatch AI — Live Video Monitoring")
    st.caption(
        "Upload any four videos. The model receives only their frames, "
        "not their filenames or class labels."
    )

    uploaded = st.file_uploader(
        "Select four unrelated video files",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        key="crimewatch_live_videos",
    )

    if not uploaded:
        st.info("Upload four videos to start the monitoring wall.")
        return

    if len(uploaded) != 4:
        st.warning(f"Please select exactly four videos. Selected: {len(uploaded)}.")
        return

    # A changed upload set resets the previous monitoring session.
    signature = tuple((f.name, f.size) for f in uploaded)
    if st.session_state.get("live_signature") != signature:
        st.session_state["live_signature"] = signature
        st.session_state.pop("live_results", None)

    if st.session_state.get("live_results") is None:
        if not st.button("Start four-feed monitoring", type="primary"):
            return

        with st.spinner("Loading the trained CNN-LSTM model..."):
            analyzer = CrimeVideoAnalyzer()

        temp_paths = []
        captures = []

        try:
            for file in uploaded:
                suffix = Path(file.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp:
                    temp.write(file.getvalue())
                    temp_paths.append(temp.name)

            for path in temp_paths:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open video: {path}")
                captures.append(cap)

            fps_values = [
                cap.get(cv2.CAP_PROP_FPS) for cap in captures
            ]
            fps_values = [fps if fps > 0 else 25.0 for fps in fps_values]

            buffers = [deque(maxlen=NUM_FRAMES) for _ in captures]
            frame_indexes = [0] * 4
            labels = ["Collecting frames"] * 4
            confidences = [0.0] * 4
            timestamps = [0.0] * 4
            active = [True] * 4

            # Separate UI placeholders for each feed.
            columns = st.columns(2)
            video_slots = []
            metric_slots = []
            status_slots = []

            for i, col in enumerate(columns * 2):
                with col:
                    st.markdown(f"**Feed {i + 1} — {uploaded[i].name}**")
                    video_slots.append(st.empty())
                    metric_slots.append(st.empty())
                    status_slots.append(st.empty())

            stop = st.button("Stop monitoring")
            if stop:
                return

            overall_status = st.empty()
            overall_status.success("Monitoring started.")

            while any(active):
                for i, cap in enumerate(captures):
                    if not active[i]:
                        continue

                    ok, frame = cap.read()
                    if not ok:
                        active[i] = False
                        status_slots[i].warning("Video ended.")
                        continue

                    frame_indexes[i] += 1
                    timestamps[i] = frame_indexes[i] / fps_values[i]
                    buffers[i].append(frame)

                    if (
                        len(buffers[i]) == NUM_FRAMES
                        and (
                            confidences[i] == 0.0
                            or frame_indexes[i] % INFERENCE_EVERY == 0
                        )
                    ):
                        label, confidence = analyzer.predict_window(
                            list(buffers[i])
                        )
                        labels[i] = label
                        confidences[i] = confidence

                    status = (
                        "INFERENCE READY"
                        if len(buffers[i]) == NUM_FRAMES
                        else f"Collecting frames {len(buffers[i])}/{NUM_FRAMES}"
                    )

                    rendered = _overlay(
                        frame,
                        labels[i],
                        confidences[i],
                        timestamps[i],
                        status,
                    )

                    video_slots[i].image(
                        rendered,
                        channels="RGB",
                        use_container_width=True,
                    )

                    metric_slots[i].markdown(
                        f"**Prediction:** {labels[i]}  \n"
                        f"**Confidence:** {confidences[i]:.2f}%  \n"
                        f"**Timestamp:** {timestamps[i]:.2f}s"
                    )

                    status_slots[i].caption(status)

                # Yield briefly so Streamlit can render updated frames.
                time.sleep(0.01)

            overall_status.info("All four videos have finished.")

        except Exception as exc:
            st.error(f"Monitoring failed: {exc}")
            st.exception(exc)

        finally:
            for cap in captures:
                cap.release()

            for path in temp_paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
