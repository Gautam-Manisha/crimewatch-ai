import os

from dotenv import load_dotenv
from google import genai

from src.rag.semantic_retriever import CrimeWatchRetriever


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found in .env"
    )

client = genai.Client(api_key=API_KEY)


def build_incident_context(events):
    """
    Convert detected incidents into text that
    the LLM can use as grounded evidence.
    """

    if not events:
        return "No suspicious incidents were detected."

    context = []

    for index, event in enumerate(
        events,
        start=1
    ):

        duration = (
            event["end_time"]
            - event["start_time"]
        )

        context.append(
            f"""
INCIDENT #{index}

Type:
{event["label"]}

Start:
{event["start_time"]:.2f} seconds

End:
{event["end_time"]:.2f} seconds

Duration:
{duration:.2f} seconds

Peak model confidence:
{event["peak_confidence"]:.2f}%
"""
        )

    return "\n".join(context)


def answer_video_question(
    question,
    events
):
    """
    Answer a user question using:
    1. Detected incidents
    2. Retrieved RAG knowledge
    """

    # --------------------------------------------------------
    # Build incident context
    # --------------------------------------------------------

    incident_context = build_incident_context(
        events
    )

    # --------------------------------------------------------
    # Retrieve relevant knowledge
    # --------------------------------------------------------

    retriever = CrimeWatchRetriever()

    results = retriever.retrieve(
        question,
        top_k=3
    )

    knowledge_context = "\n\n".join(
        result["document"]
        for result in results
    )

    # --------------------------------------------------------
    # Construct grounded prompt
    # --------------------------------------------------------

    prompt = f"""
You are CrimeWatch AI's investigation assistant.

Your task is to answer questions about a
surveillance video that has already been analyzed
by a CNN-LSTM model.

IMPORTANT RULES:

1. Use the detected incident information as the
   primary source for video-specific facts.

2. Use the retrieved knowledge as supporting
   domain context.

3. Do not invent events, timestamps, people,
   objects, causes, or evidence.

4. If the available information does not answer
   the question, explicitly say that the available
   analysis does not provide enough information.

5. Model confidence is NOT the same as model accuracy.

6. Do not make legal conclusions.

7. Keep answers concise but informative.

==================================================
DETECTED INCIDENTS
==================================================

{incident_context}

==================================================
RETRIEVED KNOWLEDGE
==================================================

{knowledge_context}

==================================================
USER QUESTION
==================================================

{question}

==================================================

Answer the user's question using only the
available evidence and retrieved context.
"""

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )

    return response.text


if __name__ == "__main__":

    sample_events = [
        {
            "label": "Fighting",
            "start_time": 10.0,
            "end_time": 18.0,
            "peak_confidence": 91.5,
        }
    ]

    question = input(
        "Ask a question about the analyzed video: "
    ).strip()

    answer = answer_video_question(
        question,
        sample_events
    )

    print("\n" + "=" * 70)
    print("CRIMEWATCH AI ANSWER")
    print("=" * 70)

    print(answer)