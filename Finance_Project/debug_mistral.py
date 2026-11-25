"""
debug_mistral.py — Deep diagnostic for Mistral API 429 / Rate Limit issues.
Run: python debug_mistral.py
"""
import os
import httpx
from dotenv import load_dotenv, dotenv_values, find_dotenv


def _mask(v: str | None) -> str:
    if not v:
        return "MISSING"
    return f"{v[:6]}...{v[-4:]}" if len(v) > 10 else "****"


def main():
    os_env_before = os.environ.get("MISTRAL_API_KEY")
    env_path = find_dotenv(usecwd=True)
    file_vals = dotenv_values(env_path) if env_path else {}
    file_key = file_vals.get("MISTRAL_API_KEY")

    load_dotenv(env_path, override=True)
    active_key = os.getenv("MISTRAL_API_KEY")

    print("=" * 70)
    print("  MISTRAL API DEEP DIAGNOSTIC")
    print("=" * 70)
    print(f".env path found        : {env_path or 'NONE'}")
    print(f"OS env before .env     : {_mask(os_env_before)}")
    print(f".env file key          : {_mask(file_key)}")
    print(f"Active key (override)  : {_mask(active_key)}")
    if os_env_before and file_key and os_env_before != file_key:
        print("WARNING: Windows OS env variable is DIFFERENT from .env! Use override=True.")

    if not active_key:
        print("ERROR: No MISTRAL_API_KEY found.")
        return

    headers = {
        "Authorization": f"Bearer {active_key.strip()}",
        "Content-Type": "application/json",
    }

    # 1. Check /v1/models
    print("\n[1] Checking GET https://api.mistral.ai/v1/models ...")
    with httpx.Client(timeout=15) as client:
        r_models = client.get("https://api.mistral.ai/v1/models", headers=headers)
        print(f"Status: {r_models.status_code}")
        if r_models.status_code == 200:
            ids = [m["id"] for m in r_models.json().get("data", [])]
            print(f"Total models accessible: {len(ids)}")
            for candidate in ["mistral-small-2506", "mistral-small-latest", "open-mistral-nemo", "ministral-8b-latest"]:
                print(f"  - {candidate:22s}: {'AVAILABLE' if candidate in ids else 'NOT IN LIST'}")
        else:
            print(f"Body: {r_models.text}")

        # 2. Test chat/completions across models and inspect exact rate-limit headers
        print("\n[2] Testing POST /v1/chat/completions + Rate-Limit Headers ...")
        for model in ["mistral-small-2506", "mistral-small-latest", "open-mistral-nemo"]:
            print(f"\n  --- Model: {model} ---")
            r = client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with OK"}],
                    "max_tokens": 5,
                },
            )
            print(f"  HTTP Status : {r.status_code}")
            rl_headers = {
                k: v for k, v in r.headers.items()
                if "ratelimit" in k.lower() or "retry" in k.lower() or "x-kong" in k.lower()
            }
            if rl_headers:
                print("  Headers     :")
                for k, v in rl_headers.items():
                    print(f"    {k}: {v}")
            print(f"  Body        : {r.text[:300]}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
