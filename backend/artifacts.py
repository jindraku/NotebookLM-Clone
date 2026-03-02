from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import requests

from backend.llm import generate_text
from storage.notebook_store import NotebookStore

REPORT_SYSTEM_PROMPT = "You produce well-structured study reports grounded in source material."
QUIZ_SYSTEM_PROMPT = "You create quizzes with answer keys grounded in study material."
PODCAST_SYSTEM_PROMPT = "You produce an educational podcast transcript between two speakers."


def _load_material(store: NotebookStore, username: str, notebook_id: str, max_chars: int = 20000) -> str:
    paths = store.notebook_paths(username, notebook_id)
    chunks: list[str] = []
    for path in sorted(paths.files_extracted.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks.append(f"## Source: {path.name}\n{text}")
        if sum(len(c) for c in chunks) > max_chars:
            break
    return "\n\n".join(chunks)[:max_chars]


def _fallback_report(material: str) -> str:
    preview = material[:2500] if material else "No source content available."
    return (
        "# Study Report\n\n"
        "## Key Themes\n- Add sources and regenerate for richer results.\n\n"
        "## Source Preview\n"
        f"{preview}\n"
    )


def _fallback_quiz(material: str) -> str:
    preview = material[:1200] if material else "No source content available."
    return (
        "# Quiz\n\n"
        "1. What are the main ideas in the uploaded materials?\n"
        "2. Which details support those ideas?\n\n"
        "# Answer Key\n\n"
        "1. Depends on your sources.\n"
        "2. Depends on your sources.\n\n"
        f"Source preview:\n{preview}\n"
    )


def _fallback_podcast(material: str) -> str:
    preview = material[:1200] if material else "No source content available."
    return (
        "# Podcast Transcript\n\n"
        "**Host A:** Today we summarize the notebook sources.\n\n"
        "**Host B:** Here's a quick preview of the material we have so far.\n\n"
        f"{preview}\n"
    )


def _parse_dialogue_turns(transcript: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Accept markdown speaker labels like: **Host A:** text
        match = re.match(r"^\**\s*(Host A|Host B)\s*\**\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if match:
            speaker = match.group(1).title()
            text = match.group(2).strip()
            if text:
                turns.append((speaker, text))

    if turns:
        return turns

    # Fallback: strip markdown and alternate speakers by sentence.
    clean = re.sub(r"[*#`_]", " ", transcript)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    speaker = "Host A"
    for sentence in sentences[:80]:
        turns.append((speaker, sentence))
        speaker = "Host B" if speaker == "Host A" else "Host A"
    return turns


def _elevenlabs_dialogue_mp3_bytes(turns: list[tuple[str, str]]) -> bytes | None:
    if not turns:
        return None

    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_a = os.getenv("ELEVENLABS_VOICE_ID_A")
    voice_b = os.getenv("ELEVENLABS_VOICE_ID_B")
    if not api_key or not voice_a or not voice_b:
        return None

    base_url = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")
    output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
    model_id = os.getenv("ELEVENLABS_DIALOGUE_MODEL", "eleven_v3")

    inputs = []
    for speaker, text in turns:
        voice_id = voice_a if speaker == "Host A" else voice_b
        inputs.append({"text": text[:800], "voice_id": voice_id})

    try:
        response = requests.post(
            f"{base_url}/v1/text-to-dialogue",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            params={"output_format": output_format},
            json={
                "inputs": inputs,
                "model_id": model_id,
            },
            timeout=180,
        )
        response.raise_for_status()
        if response.content:
            return response.content
    except Exception:
        return None

    return None


def generate_report(store: NotebookStore, username: str, notebook_id: str, extra_prompt: str = "") -> Path:
    material = _load_material(store, username, notebook_id)
    prompt = (
        "Create a markdown study report with sections: Executive Summary, Key Concepts, "
        "Open Questions, and Actionable Review Plan.\n\n"
        f"Additional user instruction: {extra_prompt or 'None'}\n\n"
        "Source material:\n"
        f"{material}"
    )
    text = generate_text(prompt, REPORT_SYSTEM_PROMPT, fallback_text=_fallback_report(material))
    return store.save_artifact_text(username, notebook_id, "report", ".md", text)


def generate_quiz(store: NotebookStore, username: str, notebook_id: str, extra_prompt: str = "") -> Path:
    material = _load_material(store, username, notebook_id)
    prompt = (
        "Create a markdown quiz with 8 questions across short-answer and multiple-choice formats. "
        "Include a final 'Answer Key' section.\n\n"
        f"Additional user instruction: {extra_prompt or 'None'}\n\n"
        "Source material:\n"
        f"{material}"
    )
    text = generate_text(prompt, QUIZ_SYSTEM_PROMPT, fallback_text=_fallback_quiz(material))
    return store.save_artifact_text(username, notebook_id, "quiz", ".md", text)


def generate_podcast(store: NotebookStore, username: str, notebook_id: str, extra_prompt: str = "") -> dict[str, str]:
    material = _load_material(store, username, notebook_id)
    prompt = (
        "Create a markdown podcast transcript as a dialogue between Host A and Host B. "
        "Length target: 4-6 minutes spoken. "
        "Keep it conversational and natural, with occasional short interjections and follow-up questions. "
        "Use one line per turn in this exact format: **Host A:** ... or **Host B:** ...\n\n"
        f"Additional user instruction: {extra_prompt or 'None'}\n\n"
        "Source material:\n"
        f"{material}"
    )
    transcript = generate_text(prompt, PODCAST_SYSTEM_PROMPT, fallback_text=_fallback_podcast(material))
    transcript_path = store.save_artifact_text(username, notebook_id, "podcast", ".md", transcript)

    audio_path = None
    turns = _parse_dialogue_turns(transcript)
    audio_bytes = _elevenlabs_dialogue_mp3_bytes(turns)
    if audio_bytes:
        try:
            audio_path = store.save_artifact_bytes(
                username,
                notebook_id,
                "podcast",
                ".mp3",
                audio_bytes,
            )
        except Exception:
            audio_path = None

    return {
        "transcript_path": str(transcript_path),
        "audio_path": str(audio_path) if audio_path else "",
    }


def list_artifacts(store: NotebookStore, username: str, notebook_id: str) -> list[dict[str, Any]]:
    return store.list_artifacts(username, notebook_id)
