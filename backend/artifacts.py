from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

from backend.llm import generate_text
from storage.notebook_store import NotebookStore

try:
    from gtts import gTTS
except Exception:  # pragma: no cover
    gTTS = None

try:
    from pydub import AudioSegment
except Exception:  # pragma: no cover
    AudioSegment = None


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
        line = re.sub(r"^\*+", "", line)
        line = re.sub(r"\*+$", "", line)
        line = line.strip()

        match = re.match(r"^(Host A|Host B)\s*:\s*(.+)$", line, flags=re.IGNORECASE)
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


def _openai_tts_mp3_bytes(text: str, voice: str) -> bytes | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
        response = client.audio.speech.create(
            model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=voice,
            input=text,
            format="mp3",
        )

        if hasattr(response, "read"):
            return response.read()
        if hasattr(response, "content"):
            return response.content
        if isinstance(response, bytes):
            return response
    except Exception:
        return None

    return None


def _synthesize_dialogue_mp3(turns: list[tuple[str, str]], output_path: Path) -> bool:
    if not turns or AudioSegment is None:
        return False

    # Distinct voices to separate hosts.
    # OpenAI voices are preferred when OPENAI_API_KEY is available; gTTS is fallback.
    openai_voice_profile = {
        "Host A": os.getenv("OPENAI_TTS_VOICE_A", "alloy"),
        "Host B": os.getenv("OPENAI_TTS_VOICE_B", "nova"),
    }
    voice_profile = {
        "Host A": {"lang": "en", "tld": "com"},
        "Host B": {"lang": "en", "tld": "co.uk"},
    }

    try:
        with tempfile.TemporaryDirectory(prefix="podcast_tts_") as tmp:
            merged = AudioSegment.silent(duration=250)
            for idx, (speaker, text) in enumerate(turns, start=1):
                segment_path = Path(tmp) / f"seg_{idx:04d}.mp3"
                clipped_text = text[:900]

                # First choice: higher-quality OpenAI TTS.
                openai_voice = openai_voice_profile.get(speaker, openai_voice_profile["Host A"])
                segment_bytes = _openai_tts_mp3_bytes(clipped_text, openai_voice)
                if segment_bytes:
                    segment_path.write_bytes(segment_bytes)
                else:
                    if gTTS is None:
                        return False
                    settings = voice_profile.get(speaker, voice_profile["Host A"])
                    tts = gTTS(text=clipped_text, lang=settings["lang"], tld=settings["tld"])
                    tts.save(str(segment_path))

                segment_audio = AudioSegment.from_file(segment_path, format="mp3")
                if speaker == "Host B":
                    segment_audio = segment_audio + 1
                else:
                    segment_audio = segment_audio - 1
                merged += segment_audio + AudioSegment.silent(duration=220)

            merged.export(output_path, format="mp3", bitrate="96k")
        return True
    except Exception:
        return False


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
    if gTTS is not None:
        try:
            temp_mp3 = Path(transcript_path).with_suffix(".mp3")
            turns = _parse_dialogue_turns(transcript)

            built_dialogue = _synthesize_dialogue_mp3(turns, temp_mp3)
            if not built_dialogue:
                # Fallback to one-voice narration if multi-speaker synthesis is unavailable.
                tts = gTTS(text=transcript[:5000], lang="en", tld="com")
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
