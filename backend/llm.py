from __future__ import annotations

import os


DEFAULT_FALLBACK = (
    "I couldn't use an external LLM provider, so this response is generated from retrieved notes only. "
    "Set OPENAI_API_KEY to enable stronger generation."
)


def _openai_generate(prompt: str, system_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = completion.choices[0].message.content
    return text.strip() if text else ""


def generate_text(prompt: str, system_prompt: str, fallback_text: str = "") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            response = _openai_generate(prompt, system_prompt)
            if response:
                return response
        except Exception:
            pass

    if fallback_text:
        return f"{fallback_text}\n\n{DEFAULT_FALLBACK}"
    return DEFAULT_FALLBACK
