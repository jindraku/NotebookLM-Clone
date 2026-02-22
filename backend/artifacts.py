from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.llm import generate_text
from storage.notebook_store import NotebookStore

try:
    from gtts import gTTS
except Exception:  # pragma: no cover
    gTTS = None


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
        "Length target: 4-6 minutes spoken.\n\n"
        f"Additional user instruction: {extra_prompt or 'None'}\n\n"
        "Source material:\n"
        f"{material}"
    )
    transcript = generate_text(prompt, PODCAST_SYSTEM_PROMPT, fallback_text=_fallback_podcast(material))
    transcript_path = store.save_artifact_text(username, notebook_id, "podcast", ".md", transcript)

    audio_path = None
    if gTTS is not None:
        try:
            tts = gTTS(text=transcript[:5000])
            temp_mp3 = Path(transcript_path).with_suffix(".mp3")
            tts.save(str(temp_mp3))
            audio_path = store.save_artifact_bytes(
                username,
                notebook_id,
                "podcast",
                ".mp3",
                Path(temp_mp3).read_bytes(),
            )
            temp_mp3.unlink(missing_ok=True)
        except Exception:
            audio_path = None

    return {
        "transcript_path": str(transcript_path),
        "audio_path": str(audio_path) if audio_path else "",
    }


def list_artifacts(store: NotebookStore, username: str, notebook_id: str) -> list[dict[str, Any]]:
    return store.list_artifacts(username, notebook_id)
