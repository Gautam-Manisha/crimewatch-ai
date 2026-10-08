from typing import List, Dict


SUSPICIOUS_LABELS = {
    "Fighting",
    "Shooting",
    "Explosion",
}


def aggregate_events(
    predictions: List[Dict],
    confidence_threshold: float = 60.0,
    merge_gap: float = 5.0,
):
    """
    Convert window-level predictions into meaningful
    suspicious incidents.

    merge_gap:
        Maximum allowed time gap between two suspicious
        windows before they are considered part of the
        same incident.
    """

    events = []

    for prediction in predictions:

        label = prediction["label"]
        confidence = prediction["confidence"]

        if confidence < confidence_threshold:
            continue

        if label not in SUSPICIOUS_LABELS:
            continue

        event = {
            "label": label,
            "start_time": prediction["start_time"],
            "end_time": prediction["end_time"],
            "peak_confidence": confidence,
        }

        events.append(event)

    if not events:
        return []

    # Sort chronologically
    events.sort(
        key=lambda x: x["start_time"]
    )

    aggregated = []

    current = events[0].copy()

    for event in events[1:]:

        time_gap = (
            event["start_time"]
            - current["end_time"]
        )

        # Same type and close enough in time
        if (
            event["label"] == current["label"]
            and time_gap <= merge_gap
        ):

            current["end_time"] = max(
                current["end_time"],
                event["end_time"]
            )

            current["peak_confidence"] = max(
                current["peak_confidence"],
                event["peak_confidence"]
            )

        else:

            aggregated.append(current)

            current = event.copy()

    aggregated.append(current)

    return aggregated


def print_events(events):

    print("\n" + "=" * 70)
    print("SUSPICIOUS EVENTS")
    print("=" * 70)

    if not events:

        print("\nNo suspicious events detected.")

        return

    for index, event in enumerate(
        events,
        start=1
    ):

        duration = (
            event["end_time"]
            - event["start_time"]
        )

        print("\n" + "-" * 70)

        print(
            f"Event #{index}"
        )

        print(
            f"Type: {event['label']}"
        )

        print(
            f"Start: "
            f"{event['start_time']:.2f}s"
        )

        print(
            f"End: "
            f"{event['end_time']:.2f}s"
        )

        print(
            f"Duration: "
            f"{duration:.2f}s"
        )

        print(
            f"Peak confidence: "
            f"{event['peak_confidence']:.2f}%"
        )