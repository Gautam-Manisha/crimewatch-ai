import json
import shutil
from pathlib import Path

import streamlit as st

from src.analysis.video_analyzer import CrimeVideoAnalyzer
from src.analysis.event_aggregator import aggregate_events
from src.analysis.evidence_extractor import (
    extract_evidence_frame,
    extract_video_segment,
)
from src.analysis.incident_recorder import save_incident_record
from src.rag.video_qa import answer_video_question

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CrimeWatch AI",
    page_icon="🎥",
    layout="wide",
)


VIDEO_OUTPUT_DIR = Path("outputs/analyzed_videos")
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

st.title("CrimeWatch AI")

st.subheader(
    "Generative Video Intelligence & Suspicious Activity Analysis"
)

st.write(
    """
CrimeWatch AI analyzes surveillance videos using a CNN-LSTM
deep learning model, identifies suspicious activities, extracts
evidence, and prepares incidents for AI-powered investigation.
"""
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System")

    st.write("**Detection Model**")
    st.write("CNN + LSTM")

    st.write("**Supported Classes**")
    st.write("- Normal")
    st.write("- Fighting")
    st.write("- Shooting")
    st.write("- Explosion")

    st.write("**Analysis Pipeline**")
    st.write("Video → CNN → LSTM → Events → Evidence → RAG → GenAI")

    st.divider()

    st.caption(
        "Automated detection is an assistance tool. "
        "All alerts require human verification."
    )


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.header("1. Upload Surveillance Video")

uploaded_file = st.file_uploader(
    "Choose a CCTV / surveillance video",
    type=["mp4", "avi", "mov", "mkv"],
)


if uploaded_file is not None:

    st.success(f"Video loaded: {uploaded_file.name}")

    video_path = VIDEO_OUTPUT_DIR / uploaded_file.name

    with open(video_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.video(str(video_path))

    st.divider()

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "Analyze Video",
        type="primary",
        use_container_width=True,
    ):

        st.header("2. Video Analysis")

        progress = st.progress(0)
        status = st.empty()

        try:

            # ------------------------------------------------
            # STEP 1 — MODEL INITIALIZATION
            # ------------------------------------------------

            status.write("Loading CNN-LSTM model...")
            progress.progress(10)

            analyzer = CrimeVideoAnalyzer()

            # ------------------------------------------------
            # STEP 2 — VIDEO ANALYSIS
            # ------------------------------------------------

            status.write("Analyzing video frames...")
            progress.progress(20)

            predictions = analyzer.analyze(str(video_path))

            progress.progress(65)

            # ------------------------------------------------
            # STEP 3 — EVENT AGGREGATION
            # ------------------------------------------------

            status.write("Aggregating suspicious events...")

            suspicious_events = aggregate_events(
                predictions,
                confidence_threshold=60.0,
                merge_gap=5.0,
            )

            progress.progress(75)

            # Save results in session state
            st.session_state["predictions"] = predictions
            st.session_state["events"] = suspicious_events
            st.session_state["video_path"] = str(video_path)

            status.write("Extracting evidence...")

            # ------------------------------------------------
            # STEP 4 — EVIDENCE EXTRACTION
            # ------------------------------------------------

            incident_results = []

            for index, event in enumerate(
                suspicious_events,
                start=1,
            ):

                incident_id = f"INCIDENT_{index:03d}"

                midpoint = (
                    event["start_time"]
                    + event["end_time"]
                ) / 2

                try:

                    frame_path = extract_evidence_frame(
                        str(video_path),
                        midpoint,
                        incident_id,
                    )

                    segment_path = extract_video_segment(
                        str(video_path),
                        event["start_time"],
                        event["end_time"],
                        incident_id,
                    )

                    record_path = save_incident_record(
                        incident_id,
                        event,
                        frame_path,
                        segment_path,
                        None,
                    )

                    incident_results.append(
                        {
                            "incident_id": incident_id,
                            "event": event,
                            "frame_path": str(frame_path),
                            "segment_path": str(segment_path),
                            "record_path": str(record_path),
                        }
                    )

                except Exception as evidence_error:

                    incident_results.append(
                        {
                            "incident_id": incident_id,
                            "event": event,
                            "error": str(evidence_error),
                        }
                    )

            progress.progress(100)

            st.session_state["incident_results"] = incident_results

            status.success("Analysis completed successfully.")

        except Exception as error:

            progress.empty()

            st.error(
                f"Analysis failed:\n\n{error}"
            )


# ============================================================
# RESULTS
# ============================================================

if "events" in st.session_state:

    events = st.session_state["events"]

    st.divider()

    st.header("3. Analysis Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Predictions",
            len(st.session_state["predictions"]),
        )

    with col2:
        st.metric(
            "Suspicious Incidents",
            len(events),
        )

    with col3:

        if events:
            max_confidence = max(
                event["peak_confidence"]
                for event in events
            )
        else:
            max_confidence = 0

        st.metric(
            "Peak Confidence",
            f"{max_confidence:.2f}%",
        )


    # ========================================================
    # NO INCIDENTS
    # ========================================================

    if not events:

        st.success(
            "No suspicious incidents were detected "
            "above the configured confidence threshold."
        )


    # ========================================================
    # INCIDENT TIMELINE
    # ========================================================

    else:

        st.header("4. Incident Timeline")

        for index, event in enumerate(
            events,
            start=1,
        ):

            incident_id = f"INCIDENT_{index:03d}"

            st.markdown(
                f"### 🚨 {incident_id}"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.write("**Type**")
                st.write(event["label"])

            with col2:
                st.write("**Start**")
                st.write(
                    f'{event["start_time"]:.2f}s'
                )

            with col3:
                st.write("**End**")
                st.write(
                    f'{event["end_time"]:.2f}s'
                )

            with col4:
                st.write("**Confidence**")
                st.write(
                    f'{event["peak_confidence"]:.2f}%'
                )


            # Find evidence
            incident_data = None

            for result in st.session_state.get(
                "incident_results",
                [],
            ):

                if result["incident_id"] == incident_id:
                    incident_data = result
                    break


            if incident_data:

                evidence_col, report_col = st.columns(
                    [1, 1]
                )

                # ------------------------------------------------
                # EVIDENCE
                # ------------------------------------------------

                with evidence_col:

                    st.subheader("Evidence")

                    frame_path = incident_data.get(
                        "frame_path"
                    )

                    segment_path = incident_data.get(
                        "segment_path"
                    )

                    if frame_path and Path(
                        frame_path
                    ).exists():

                        st.image(
                            frame_path,
                            caption=(
                                f"{incident_id} "
                                "Representative Frame"
                            ),
                        )

                    if segment_path and Path(
                        segment_path
                    ).exists():

                        st.video(segment_path)


                # ------------------------------------------------
                # AI REPORT
                # ------------------------------------------------

                with report_col:

                    st.subheader(
                        "AI Incident Report"
                    )

                    st.info(
                        """
AI report generation is connected to the
RAG + Gemini intelligence layer.

If the external LLM quota is unavailable,
the detection and evidence pipeline still
works normally.
"""
                    )

                    st.write(
                        "Report generation will be "
                        "enabled here when the "
                        "LLM service is available."
                    )


            st.divider()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "CrimeWatch AI — CNN-LSTM + RAG + Generative AI"
)
# ============================================================
# VIDEO Q&A / INVESTIGATION ASSISTANT
# ============================================================

if "events" in st.session_state:

    st.divider()

    st.header("5. Investigation Assistant")

    st.write(
        """
Ask questions about the analyzed video.
The assistant uses detected incidents and
retrieved CrimeWatch knowledge to provide
grounded answers.
"""
    )

    question = st.text_input(
        "Ask a question",
        placeholder=(
            "Example: When did the suspicious "
            "activity occur?"
        ),
    )

    if st.button(
        "Ask CrimeWatch AI",
        use_container_width=True,
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "Analyzing your question..."
            ):

                try:

                    answer = answer_video_question(
                        question,
                        st.session_state["events"]
                    )

                    st.subheader(
                        "CrimeWatch AI Answer"
                    )

                    st.write(answer)

                except Exception as error:

                    error_text = str(error)

                    if (
                        "429" in error_text
                        or
                        "RESOURCE_EXHAUSTED"
                        in error_text
                    ):

                        st.warning(
                            """
The video analysis pipeline is working,
but the external Gemini API quota is
currently unavailable.

Detection, incident aggregation, and
evidence extraction continue to work
independently.

The Q&A assistant will work again when
the API quota becomes available.
"""
                        )

                    else:

                        st.error(
                            f"Q&A failed: {error}"
                        )