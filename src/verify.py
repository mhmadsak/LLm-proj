import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"

def verify_facts_with_context(fact, context):
    """Use DeepSeek API to verify if fact is supported by context."""
    if not context:
        return {"supported": False, "reason": "No context"}

    prompt = f"Fact: {fact}\nContext: {context}\n\nAnswer with 'SUPPORTED' or 'NOT SUPPORTED'."

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
    }

    r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=20)
    if r.status_code != 200:
        return {"supported": False, "reason": "API error"}

    text = r.json()["choices"][0]["message"]["content"].strip().upper()
    return {"supported": "SUPPORTED" in text}