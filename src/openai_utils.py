import json
import re
from typing import Any

from openai import OpenAI

from src.config import load_settings


def get_openai_client() -> tuple[OpenAI, str]:
    api_key, model = load_settings()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and fill OPENAI_API_KEY."
        )
    return OpenAI(api_key=api_key), model


def chat_completion(prompt: str, temperature: float = 0.0) -> str:
    client, model = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def parse_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, list):
        raise ValueError("OpenAI response JSON is not an array.")
    return [item for item in parsed if isinstance(item, dict)]
