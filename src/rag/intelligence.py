from src.rag.semantic_retriever import CrimeWatchRetriever
from src.rag.incident_report import generate_incident_report


def build_event_query(event):
    """
    Convert a detected surveillance event into
    a semantic retrieval query.
    """

    return f"""
A surveillance video analysis system detected
a potential {event["label"]} event.

The event occurred from
{event["start_time"]:.2f} seconds to
{event["end_time"]:.2f} seconds.

The model confidence was
{event["peak_confidence"]:.2f} percent.

Retrieve relevant information about:

- surveillance interpretation
- incident handling
- evidence preservation
- model interpretation
- limitations
- human verification

The retrieved information will be used to
generate an automated incident report.
"""


def generate_event_report(event):

    # Load semantic RAG system
    retriever = CrimeWatchRetriever()

    # Build semantic search query
    query = build_event_query(event)

    # Retrieve the three most relevant documents
    results = retriever.retrieve(
        query,
        top_k=3
    )

    # Extract retrieved documents
    context_parts = []

    for result in results:
        context_parts.append(result["document"])

    retrieved_context = "\n\n".join(context_parts)

    # Generate the AI incident report
    report = generate_incident_report(
        event,
        retrieved_context
    )

    return {
        "event": event,
        "retrieved_context": retrieved_context,
        "report": report
    }


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    test_event = {
        "label": "Fighting",
        "start_time": 42.5,
        "end_time": 57.2,
        "peak_confidence": 87.4
    }

    result = generate_event_report(test_event)

    print("\n" + "=" * 70)
    print("RETRIEVED RAG CONTEXT")
    print("=" * 70)

    print(result["retrieved_context"])

    print("\n" + "=" * 70)
    print("AI INCIDENT REPORT")
    print("=" * 70)

    print(result["report"])