import os

from dotenv import load_dotenv
from google import genai


# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

# Create Gemini client
client = genai.Client(api_key=API_KEY)


def generate_incident_report(event, retrieved_context):

    prompt = f"""
You are the intelligence and reporting layer
of CrimeWatch AI, an AI-powered video
surveillance analysis system.

You are given:
1. An automated video detection event.
2. Relevant context retrieved from the CrimeWatch knowledge base.

Your job is to generate a concise, professional incident report.

IMPORTANT:
- Do not claim the incident is legally confirmed.
- Clearly state that this is an automated model detection.
- Do not invent facts.
- Do not invent people, weapons, locations, motives, or actions.
- Preserve the model confidence exactly.
- Recommend human verification.
- Use retrieved context only when relevant.

DETECTED EVENT:
Event type: {event["label"]}
Start time: {event["start_time"]:.2f} seconds
End time: {event["end_time"]:.2f} seconds
Peak model confidence: {event["peak_confidence"]:.2f}%

RETRIEVED KNOWLEDGE:
{retrieved_context}

Generate the report using exactly these sections:

INCIDENT TYPE
TIMESTAMP
MODEL CONFIDENCE
AUTOMATED ASSESSMENT
RELEVANT CONTEXT
RECOMMENDED ACTION
IMPORTANT LIMITATION
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )

    return response.text


# Test the incident report generator directly
if __name__ == "__main__":

    test_event = {
        "label": "Fighting",
        "start_time": 42.5,
        "end_time": 57.2,
        "peak_confidence": 87.4
    }

    test_context = """
Fighting may involve physical confrontation between individuals.
Automated detections should be verified using the original footage.
The system's prediction is an automated alert and is not a legal conclusion.
"""

    report = generate_incident_report(
        test_event,
        test_context
    )

    print("\n" + "=" * 70)
    print("AI INCIDENT REPORT")
    print("=" * 70)
    print(report)