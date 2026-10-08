from pathlib import Path
import cv2
import torch

from src.models.cnn_feature_extractor import CNNFeatureExtractor
from src.models.cnn_lstm import CNNLSTM
from src.preprocessing.frame_processor import preprocess_frames
from src.analysis.event_aggregator import (
    aggregate_events,
    print_events,
)
from src.rag.intelligence import generate_event_report
from src.analysis.evidence_extractor import (
    extract_evidence_frame,
    extract_video_segment,
)

from src.analysis.incident_recorder import (
    save_incident_record,
)
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

MODEL_PATH = "models/cnn_lstm_best.pth"

NUM_FRAMES = 20
WINDOW_STRIDE = 10

LABELS = [
    "Normal",
    "Fighting",
    "Shooting",
    "Explosion",
]


class CrimeVideoAnalyzer:

    def __init__(self):

        print("=" * 60)
        print("CRIMEWATCH AI - VIDEO ANALYZER")
        print("=" * 60)

        print("Device:", DEVICE)

        # CNN
        self.cnn = CNNFeatureExtractor().to(DEVICE)
        self.cnn.eval()

        # LSTM
        self.lstm = CNNLSTM()

        

        state_dict = torch.load(
                MODEL_PATH,
                map_location="cpu",
               weights_only=True
               )

        self.lstm.load_state_dict(state_dict)

        self.lstm.to(DEVICE)
        self.lstm.eval()

    def get_video_info(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        duration = (
            frame_count / fps
            if fps > 0
            else 0
        )

        cap.release()

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
        }

    def predict_window(self, frames):

        processed_frames = preprocess_frames(frames)

        processed_frames = processed_frames.to(DEVICE)

        with torch.no_grad():

            # CNN
            features = self.cnn(processed_frames)

            # Add batch dimension
            features = features.unsqueeze(0)

            # LSTM
            logits = self.lstm(features)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = (
                probabilities[0][predicted_class]
                .item()
                * 100
            )

        return (
            LABELS[predicted_class],
            confidence
        )

    def analyze(self, video_path):

        video_path = str(Path(video_path))

        info = self.get_video_info(video_path)

        print()
        print("VIDEO INFORMATION")
        print("-" * 40)
        print("Video:", video_path)
        print("FPS:", info["fps"])
        print("Frames:", info["frame_count"])
        print(
            "Duration:",
            round(info["duration"], 2),
            "seconds"
        )

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        events = []

        frame_buffer = []

        frame_index = 0

        print()
        print("Starting video analysis...")
        print("-" * 60)

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_buffer.append(frame)
            frame_index += 1

            # Wait until we have 20 frames
            if len(frame_buffer) < NUM_FRAMES:
                continue

            # Analyze current temporal window
            label, confidence = self.predict_window(
                frame_buffer
            )

            start_frame = frame_index - NUM_FRAMES
            end_frame = frame_index

            start_time = (
                start_frame / info["fps"]
                if info["fps"] > 0
                else 0
            )

            end_time = (
                end_frame / info["fps"]
                if info["fps"] > 0
                else 0
            )

            event = {
                "start_time": start_time,
                "end_time": end_time,
                "label": label,
                "confidence": confidence,
            }

            events.append(event)

            print(
                f"{start_time:7.2f}s - "
                f"{end_time:7.2f}s | "
                f"{label:10s} | "
                f"{confidence:.2f}%"
            )

            # Slide the temporal window
            frame_buffer = frame_buffer[
                WINDOW_STRIDE:
            ]

        cap.release()

        print()
        print("=" * 60)
        print("VIDEO ANALYSIS COMPLETE")
        print("=" * 60)

        return events


if __name__ == "__main__":

    analyzer = CrimeVideoAnalyzer()

    video_path = input("Enter video path: ").strip()

    events = analyzer.analyze(video_path)

    suspicious_events = aggregate_events(events)

    print_events(suspicious_events)

    print("\n" + "=" * 70)
    print("EXTRACTING INCIDENT EVIDENCE")
    print("=" * 70)

    for index, event in enumerate(
        suspicious_events,
        start=1
    ):

        incident_id = f"INCIDENT_{index:03d}"

        print("\n" + "-" * 70)
        print(incident_id)
        print("-" * 70)

        print(
            f"Type: {event['label']}"
        )

        print(
            f"Time: "
            f"{event['start_time']:.2f}s → "
            f"{event['end_time']:.2f}s"
        )

        print(
            f"Confidence: "
            f"{event['peak_confidence']:.2f}%"
        )

        try:

            midpoint = (
                event["start_time"]
                + event["end_time"]
            ) / 2

            frame_path = extract_evidence_frame(
                video_path,
                midpoint,
                incident_id
            )

            segment_path = extract_video_segment(
                video_path,
                event["start_time"],
                event["end_time"],
                incident_id
            )

            incident_record = save_incident_record(
                incident_id,
                event,
                frame_path,
                segment_path,
                None
            )

            print(
                f"Evidence frame: {frame_path}"
            )

            print(
                f"Video segment: {segment_path}"
            )

            print(
                f"Incident record: {incident_record}"
            )

        except Exception as error:

            print(
                f"Could not extract evidence: {error}"
            )
        print(e)