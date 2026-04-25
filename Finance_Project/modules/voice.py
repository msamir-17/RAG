import os
from groq import Groq
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

def transcribe_audio(audio_bytes):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Whisper needs a named file, so we wrap bytes in BytesIO with a name
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "input.wav" 

    finance_hints = "audit report, statement, IFSC, debit, credit, balance, transactions, Swiggy, Zomato, Uber"

    try:
        translation = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            prompt=finance_hints,
            response_format="json"
        )
        return translation.text
    except Exception as e:
        return f"Error transcribing: {e}"