import cv2
import torch
import numpy as np
from collections import deque

from src.models.cnn_feature_extractor import CNNFeatureExtractor
from src.models.cnn_lstm import CNNLSTM


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

MODEL_PATH = "models/cnn_lstm_best.pth"

NUM_FRAMES = 20
IMAGE_SIZE = 224

LABELS = [
    "Normal",
    "Fighting",
    "Shooting",
    "Explosion",
]


# ============================================================
# FRAME PREPROCESSING
# ============================================================

def preprocess_frame(frame):

    # OpenCV BGR -> RGB
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    frame = cv2.resize(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Same normalization used during training
    frame = frame.astype(
        np.float32
    ) / 255.0

    # HWC -> CHW
    frame = np.transpose(
        frame,
        (2, 0, 1)
    )

    return torch.from_numpy(frame)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")
print("Device:", DEVICE)

cnn = CNNFeatureExtractor()
cnn = cnn.to(DEVICE)
cnn.eval()

lstm = CNNLSTM()

lstm.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

lstm = lstm.to(DEVICE)
lstm.eval()

print("Models loaded successfully.")


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam. "
        "Check macOS camera permissions."
    )


frame_buffer = deque(
    maxlen=NUM_FRAMES
)

prediction = "Collecting frames..."
confidence = 0.0

frame_count = 0


print()
print("===================================")
print("CRIMEWATCH AI LIVE DETECTION")
print("===================================")
print("Press Q to quit.")


# ============================================================
# LIVE LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    frame_count += 1

    # Add frame to buffer
    frame_buffer.append(frame.copy())


    # --------------------------------------------------------
    # Run prediction whenever we have 20 frames
    # --------------------------------------------------------

    if len(frame_buffer) == NUM_FRAMES:

        processed_frames = []

        for buffered_frame in frame_buffer:

            processed = preprocess_frame(
                buffered_frame
            )

            processed_frames.append(
                processed
            )

        # Shape:
        # (20, 3, 224, 224)

        input_tensor = torch.stack(
            processed_frames
        )

        input_tensor = input_tensor.to(
            DEVICE
        )


        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        with torch.no_grad():

            features = cnn(
                input_tensor
            )

            # features:
            # (20, 512)

            features = features.unsqueeze(0)

            # (1, 20, 512)


            # ------------------------------------------------
            # LSTM
            # ------------------------------------------------

            logits = lstm(
                features
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = (
                probabilities[0][
                    predicted_class
                ].item()
                * 100
            )

            prediction = LABELS[
                predicted_class
            ]


    # ========================================================
    # DISPLAY
    # ========================================================

    status = (
        "NORMAL"
        if prediction == "Normal"
        else "SUSPICIOUS ACTIVITY"
    )

    # Main prediction
    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0)
        if prediction == "Normal"
        else (0, 0, 255),
        2,
    )

    # Confidence
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # Status
    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0)
        if prediction == "Normal"
        else (0, 0, 255),
        2,
    )

    # Device
    cv2.putText(
        frame,
        f"Device: {DEVICE}",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    # Frames collected
    cv2.putText(
        frame,
        f"Frames: {len(frame_buffer)}/{NUM_FRAMES}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    cv2.imshow(
        "CrimeWatch AI - Live Detection",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print()
print("Live detection stopped.")