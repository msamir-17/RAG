"""
voice.py — FIX BUG 2: Lazy-import groq and langchain_mistralai.

BEFORE: groq + langchain_mistralai loaded at module import time = 4,582 ms cold startup cost.
AFTER:  Both imported only when user clicks the microphone button = 0 ms cold startup cost.
"""
import hashlib
import os
from io import BytesIO


def get_audio_hash(audio_bytes: bytes) -> str:
    """Generates a unique 16-character fingerprint for the audio data."""
    if not audio_bytes:
        return ""
    return hashlib.sha256(audio_bytes).hexdigest()[:16]


def transcribe_audio(audio_bytes: bytes) -> str | None:
    """
    FIX BUG 2: Groq client created lazily — only when mic button is used.
    groq import cost = 1,399 ms. Lazy import = 0 ms on cold start.
    """
    from groq import Groq  # Lazy import
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "input.wav"

    try:
        result = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            language="en",
            prompt=(
                "This is a finance assistant. Commands include: show forecast, open budget, "
                "generate audit report, transactions, balance, expenses"
            ),
            response_format="text",
            temperature=0
        )
        return str(result).strip()
    except Exception as e:
        return f"Error: {e}"


def classify_intent(transcript: str):
    """Uses Groq to classify spoken intent into chat, audit, budget, forecast, or navigate."""
    from modules.schema import IntentResponse
    from modules.advicor import _groq_chat

    system_prompt = """You are an intent classifier for a finance app.
    Classify the query into: chat, audit, budget, forecast, or navigate.
    - 'chat': Questions about data.
    - 'audit': Wants to generate/see the full report.
    - 'budget': Wants to set/see goals.
    - 'forecast': Wants to see future predictions.
    - 'navigate': Switching pages without a specific question.
    Return JSON with keys: intent, confidence (float), rephrased (str).
    """

    try:
        raw_json = _groq_chat(f"{system_prompt}\n\nQuery: {transcript}", temperature=0, json_mode=True)
        return IntentResponse.model_validate_json(raw_json)
    except Exception:
        return IntentResponse(intent="chat", confidence=0.5, rephrased=transcript)


def normalize_transcript(text: str) -> str:
    t = text.lower()

    if any(w in t for w in ["forecast", "future", "trend", "power"]):
        return "show forecast"
    if any(w in t for w in ["budget", "fun budget"]):
        return "open budget"
    if any(w in t for w in ["report", "audit"]):
        return "generate audit report"

    return text
