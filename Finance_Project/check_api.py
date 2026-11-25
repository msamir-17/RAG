"""
check_api.py — Quick CLI diagnostic to verify Google Gemini & Groq API connectivity and responses.
Run: python check_api.py
"""
import os
import time
from dotenv import load_dotenv, find_dotenv


def _mask(key: str | None) -> str:
    if not key:
        return "MISSING"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def main():
    load_dotenv(find_dotenv(usecwd=True), override=True)

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    print("=" * 64)
    print("  API CONNECTIVITY & RESPONSE CHECK (HYBRID: GEMINI + GROQ)")
    print("=" * 64)
    print(f"GEMINI_API_KEY  : {_mask(gemini_key)}")
    print(f"GROQ_API_KEY    : {_mask(groq_key)}")
    print("-" * 64)

    # 1. Test Google Gemini API (Audit & Forecast)
    print("\n[1/2] Testing Google Gemini (Audit & Forecast Provider)...")
    if not gemini_key:
        print("  SKIP: GEMINI_API_KEY not found in .env")
    else:
        try:
            t0 = time.perf_counter()
            from modules.advicor import _gemini_generate
            reply = _gemini_generate("Reply with exactly: GEMINI_OK", temperature=0)
            ms = (time.perf_counter() - t0) * 1000
            print(f"  STATUS   : OK ({ms:.0f} ms)")
            print(f"  RESPONSE : {reply.strip()}")
        except Exception as e:
            print(f"  STATUS   : FAILED")
            print(f"  ERROR    : {e}")

    # 2. Test Groq API (Chat & Voice)
    print("\n[2/2] Testing Groq API (Chat & Voice Provider)...")
    if not groq_key:
        print("  SKIP: GROQ_API_KEY not found in .env")
    else:
        try:
            from groq import Groq
            from modules.advicor import _pick_groq_chat_model, _groq_chat

            client = Groq(api_key=groq_key.strip())
            t0 = time.perf_counter()
            chosen_model = _pick_groq_chat_model(client)
            reply = _groq_chat("Reply with exactly: GROQ_OK", temperature=0)
            ms = (time.perf_counter() - t0) * 1000
            print(f"  MODEL    : {chosen_model}")
            print(f"  STATUS   : OK ({ms:.0f} ms)")
            print(f"  RESPONSE : {reply.strip()}")
        except Exception as e:
            print(f"  STATUS   : FAILED")
            print(f"  ERROR    : {e}")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
