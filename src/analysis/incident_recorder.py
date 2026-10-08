import json
from pathlib import Path


OUTPUT_DIR = Path("outputs/events")


def save_incident_record(
    incident_id,
    event,
    frame_path,
    segment_path,
    report=None
):
    """
    Save all information about a detected incident
    into a structured JSON record.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    incident = {
        "incident_id": incident_id,

        "incident_type": event["label"],

        "start_time": event["start_time"],

        "end_time": event["end_time"],

        "duration": (
            event["end_time"]
            - event["start_time"]
        ),

        "peak_confidence": event[
            "peak_confidence"
        ],

        "evidence": {
            "representative_frame": str(
                frame_path
            ),
            "video_segment": str(
                segment_path
            )
        },

        "ai_report": report
    }

    output_path = (
        OUTPUT_DIR /
        f"{incident_id}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            incident,
            file,
            indent=4
        )

    return output_path