import hashlib
from langchain_mistralai import ChatMistralAI
from modules.schema import IntentResponse
import os
import hashlib
from groq import Groq
from io import BytesIO
from langchain_mistralai import ChatMistralAI
from modules.schema import IntentResponse

# --- Phase 1B: Semantic Classification ---


# --- Helper for Phase 1A ---


def get_audio_hash(audio_bytes: bytes) -> str:
    """Generates a unique 16-character fingerprint for the audio data."""
    if not audio_bytes:
        return ""
    return hashlib.sha256(audio_bytes).hexdigest()[:16]


# --- 2. THE TRANSCRIPTION (Speech to Text) ---
def transcribe_audio(audio_bytes: bytes) -> str | None:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "input.wav" 

    try:
        result = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            language="en",
            prompt="This is a finance assistant. Commands include: show forecast, open budget, generate audit report, transactions, balance, expenses",
            response_format="text", # Keeps it simple and fast
            temperature=0
        )
        return str(result).strip()
    except Exception as e:
        return f"Error: {e}"

def classify_intent(transcript: str) -> IntentResponse:
    """Uses Mistral to understand if the user wants to chat, navigate, or report."""
    llm = ChatMistralAI(model="mistral-small-2506", temperature=0)
    structured_llm = llm.with_structured_output(IntentResponse)
    
    system_prompt = """You are an intent classifier for a finance app. 
    Classify the query into: chat, audit, budget, forecast, or navigate.
    - 'chat': Questions about data.
    - 'audit': Wants to generate/see the full report.
    - 'budget': Wants to set/see goals.
    - 'forecast': Wants to see future predictions.
    - 'navigate': Switching pages without a specific question.
    """
    
    try:
        return structured_llm.invoke(f"{system_prompt}\n\nQuery: {transcript}")
    except:
        # Fallback if AI fails
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