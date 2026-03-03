from __future__ import annotations

import os
import re


DEFAULT_FALLBACK = (
    "Note: running in fallback mode from retrieved notes only. "
    "Add GROQ_API_KEY or OPENAI_API_KEY for full model responses."
)


def _clean_model_text(text: str) -> str:
    # Remove leaked internal/special tokens from some model providers.
    cleaned = re.sub(r"<\|[^|>]*\|>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _chat_completion(
    prompt: str,
    system_prompt: str,
    *,
    api_key: str,
    model: str,
    base_url: str | None = None,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = completion.choices[0].message.content
    return _clean_model_text(text) if text else ""


def generate_text(prompt: str, system_prompt: str, fallback_text: str = "") -> str:
    provider_errors: list[str] = []

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            response = _chat_completion(
                prompt,
                system_prompt,
                api_key=groq_api_key,
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            )
            if response:
                return response
        except Exception as exc:
            provider_errors.append(f"GROQ error: {str(exc)[:220]}")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            response = _chat_completion(
                prompt,
                system_prompt,
                api_key=openai_api_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            )
            if response:
                return response
        except Exception as exc:
            provider_errors.append(f"OPENAI error: {str(exc)[:220]}")

    error_hint = ""
    if provider_errors:
        error_hint = "\n\nProvider details: " + " | ".join(provider_errors)

    if fallback_text:
        return f"{fallback_text}\n\n{DEFAULT_FALLBACK}{error_hint}"
    return f"{DEFAULT_FALLBACK}{error_hint}"
