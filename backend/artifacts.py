from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import requests

try:
    from gtts import gTTS
except Exception:  # pragma: no cover
    gTTS = None

from backend.llm import generate_text
from storage.notebook_store import NotebookStore

REPORT_SYSTEM_PROMPT = "You produce well-structured study reports grounded in source material."
QUIZ_SYSTEM_PROMPT = "You create quizzes with answer keys grounded in study material."
PODCAST_SYSTEM_PROMPT = "You produce an educational podcast transcript between two speakers."
ELEVENLABS_MAX_CHARS = int(os.getenv("ELEVENLABS_MAX_CHARS", "2000"))


def _normalize_source_text(text: str) -> str:
    # Flatten broken line wraps from OCR/PDF/TXT so prompts get readable context.
    cleaned = re.sub(r"\s+", " ", text).strip()
    # Re-introduce sentence breaks for readability.
    cleaned = re.sub(r"([.!?])\s+", r"\1\n", cleaned)
    return cleaned


def _load_material(store: NotebookStore, username: str, notebook_id: str, max_chars: int = 20000) -> str:
    paths = store.notebook_paths(username, notebook_id)
    chunks: list[str] = []
    for path in sorted(paths.files_extracted.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = _normalize_source_text(text)
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
        "# Study Quiz\n\n"
        "## Questions\n\n"
        "1. Which statement best summarizes the main idea from the uploaded sources?\n"
        "A. An unrelated concept\n"
        "B. A source-grounded summary\n"
        "C. A random guess\n"
        "D. None of the above\n\n"
        "2. Which evidence from the sources most strongly supports that main idea?\n"
        "A. A direct detail from the notes\n"
        "B. An unrelated anecdote\n"
        "C. A personal opinion only\n"
        "D. No evidence needed\n\n"
        "## Answer Key\n\n"
        "1. B - This reflects the core source-grounded summary.\n"
        "2. A - Strong answers cite concrete evidence from the uploaded material.\n\n"
        "## Source Snippet\n\n"
        f"{preview}\n"
    )


def _fallback_podcast(material: str) -> str:
    preview = material[:2000] if material else "No source content available."
    return (
        "# Podcast Transcript\n\n"
        "**Host A:** Today we summarize the notebook sources.\n\n"
        "**Host B:** Here's a quick preview of the material we have so far.\n\n"
        f"{preview}\n"
    )


def _format_report_markdown(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Many models return markdown as one long line. Re-introduce explicit markdown breaks.
    cleaned = re.sub(r"\s+(#{1,3}\s)", r"\n\n\1", cleaned)
    cleaned = re.sub(r"\s+(-\s)", r"\n\1", cleaned)
    cleaned = re.sub(r"\s+(\d+\.\s)", r"\n\1", cleaned)
    cleaned = re.sub(r"(\.)\s+(#{1,3}\s)", r"\1\n\n\2", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    if not cleaned.startswith("# "):
        cleaned = f"# Study Report\n\n{cleaned}"

    # Ensure heading separation for readability.
    cleaned = re.sub(r"\n(## )", r"\n\n\1", cleaned)
    cleaned = re.sub(r"\n(### )", r"\n\n\1", cleaned)

    # Ensure list blocks are not glued to heading lines.
    cleaned = re.sub(r"(#+[^\n]+)\n(- )", r"\1\n\n\2", cleaned)
    cleaned = re.sub(r"(#+[^\n]+)\n(\d+\. )", r"\1\n\n\2", cleaned)

    # If no bullet/numbered lists were produced, add a basic structure.
    has_bullets = bool(re.search(r"^\s*[-*]\s+", cleaned, flags=re.MULTILINE))
    has_numbered = bool(
        re.search(r"^\s*\d+\.\s+", cleaned, flags=re.MULTILINE))
    if not has_bullets:
        cleaned += "\n\n## Key Takeaways\n- Add clearer bullet takeaways.\n- Add evidence linked to sources."
    if not has_numbered:
        cleaned += (
            "\n\n## Worked Examples\n"
            "1. Example scenario from the sources.\n"
            "2. Step-by-step interpretation.\n"
            "3. Practical takeaway."
        )
    return cleaned.strip() + "\n"


def _format_quiz_markdown(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Split into body and answer-key sections.
    answer_key_match = re.search(
        r"(?im)^\s*(?:#{1,3}\s*)?answer\s*key\s*:?\s*$", cleaned)
    if answer_key_match:
        question_text = cleaned[: answer_key_match.start()].strip()
        answer_text = cleaned[answer_key_match.end():].strip()
    else:
        parts = re.split(r"(?i)\banswer\s*key\b\s*:?", cleaned, maxsplit=1)
        question_text = parts[0].strip()
        answer_text = parts[1].strip() if len(parts) > 1 else ""

    # Remove leading title/heading clutter from question section.
    question_text = re.sub(r"(?im)^\s*#.*$", "", question_text)
    question_text = re.sub(r"(?im)^\s*questions\s*:?\s*$", "", question_text)
    # Some providers flatten markdown into one line; force separators before inline "Question N:" markers.
    question_text = re.sub(
        r"\s*(?:#{1,3}\s*)?(Question\s*\d+\s*[:\.-])",
        r"\n\1",
        question_text,
        flags=re.IGNORECASE,
    )
    # Also force separators before inline numbered questions like " 2. ..."
    question_text = re.sub(r"\s+(\d+\s*[\.\):\-]\s+)", r"\n\1", question_text)
    question_text = re.sub(r"\n{3,}", "\n\n", question_text).strip()

    # Build normalized question blocks: question line + A/B/C/D each on separate line.
    question_blocks: list[str] = []
    q_matches = list(
        re.finditer(
            r"(?is)(?:^|\n)\s*(?:question\s*)?(\d+)\s*[\.\):\-]\s*(.*?)(?=(?:\n\s*(?:question\s*)?\d+\s*[\.\):\-])|\Z)",
            question_text,
        )
    )
    for m in q_matches:
        q_num = m.group(1)
        body = m.group(2).strip()
        body = re.sub(r"\s+", " ", body)

        first_opt = re.search(r"(?:^|\s)([A-Da-d])[.)]\s*", body)
        if first_opt:
            q_prompt = body[: first_opt.start()].strip(" -:\t")
            opt_blob = body[first_opt.start():].strip()
        else:
            q_prompt = body
            opt_blob = ""

        options: dict[str, str] = {}
        if opt_blob:
            for opt in re.finditer(r"([A-Da-d])[.)]\s*(.*?)(?=(?:\s+[A-Da-d][.)]\s)|\Z)", opt_blob):
                key = opt.group(1).upper()
                value = re.sub(r"\s+", " ", opt.group(2)).strip(" .-")
                if value:
                    options[key] = value

        # Use explicit HTML breaks to avoid markdown line-collapsing in Gradio preview.
        lines = [f"{q_num}. {q_prompt or 'Question text missing.'}<br>"]
        for key in ["A", "B", "C", "D"]:
            option_value = options.get(key, f"Option {key}")
            lines.append(f"{key}. {option_value}<br>")
        question_blocks.append("\n".join(lines))

    if not question_blocks:
        # Secondary fallback: build questions from question-style lines even if numbering was irregular.
        line_candidates = []
        for line in question_text.splitlines():
            l = line.strip()
            if not l:
                continue
            if "?" in l and len(l) > 12:
                line_candidates.append(l)
        if line_candidates:
            for i, q in enumerate(line_candidates[:8], start=1):
                question_blocks.append(
                    f"{i}. {q}<br>\n"
                    "A. Option A<br>\n"
                    "B. Option B<br>\n"
                    "C. Option C<br>\n"
                    "D. Option D<br>"
                )
        else:
            question_blocks.append(
                "1. Add source-grounded question text.<br>\n"
                "A. Option A<br>\n"
                "B. Option B<br>\n"
                "C. Option C<br>\n"
                "D. Option D<br>"
            )

    # Normalize answer key to one numbered line per entry.
    answer_map: dict[str, str] = {}
    for am in re.finditer(r"(?im)(?<!\d)(\d+)\s*[\.\):\-]\s*([A-Da-d])\b", answer_text):
        answer_map[am.group(1)] = am.group(2).upper()

    if not answer_map:
        # Fallback: infer A for each parsed question if model omitted answer key formatting.
        for i, _ in enumerate(question_blocks, start=1):
            answer_map[str(i)] = "A"

    answer_lines: list[str] = []
    for i in range(1, len(question_blocks) + 1):
        key = answer_map.get(str(i), "A")
        answer_lines.append(f"{i}. {key}")

    output = (
        "# Study Quiz\n\n"
        "## Questions\n\n"
        + "\n\n".join(question_blocks)
        + "\n\n## Answer Key\n\n"
        + "\n".join(answer_lines)
        + "\n"
    )
    return output


def _parse_dialogue_turns(transcript: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Accept markdown speaker labels like: **Host A:** text
        match = re.match(
            r"^\**\s*(Host A|Host B)\s*\**\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if match:
            speaker = match.group(1).title()
            text = match.group(2).strip()
            if text:
                turns.append((speaker, text))

    if turns:
        return turns

    # Fallback: strip markdown and alternate speakers by sentence.
    clean = re.sub(r"[*#`_]", " ", transcript)
    sentences = [s.strip() for s in re.split(
        r"(?<=[.!?])\s+", clean) if s.strip()]
    speaker = "Host A"
    for sentence in sentences[:80]:
        turns.append((speaker, sentence))
        speaker = "Host B" if speaker == "Host A" else "Host A"
    return turns


def _truncate_turns_for_tts(turns: list[tuple[str, str]], max_chars: int) -> list[tuple[str, str]]:
    if max_chars <= 0:
        return turns

    total = 0
    trimmed: list[tuple[str, str]] = []
    for speaker, text in turns:
        budget = max_chars - total
        if budget <= 0:
            break
        chunk = text[:budget].strip()
        if not chunk:
            continue
        trimmed.append((speaker, chunk))
        total += len(chunk)
    return trimmed


def _elevenlabs_dialogue_mp3_bytes(turns: list[tuple[str, str]]) -> tuple[bytes | None, str]:
    if not turns:
        return None, "No dialogue turns were parsed from the transcript."

    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_a = os.getenv("ELEVENLABS_VOICE_ID_A")
    voice_b = os.getenv("ELEVENLABS_VOICE_ID_B")
    if not api_key or not voice_a or not voice_b:
        return None, "Missing ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID_A / ELEVENLABS_VOICE_ID_B."

    base_url = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")
    output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
    model_id = os.getenv("ELEVENLABS_DIALOGUE_MODEL", "eleven_v3")

    limited_turns = _truncate_turns_for_tts(turns, ELEVENLABS_MAX_CHARS)
    if not limited_turns:
        return None, "No dialogue content available within TTS character budget."

    inputs = []
    for speaker, text in limited_turns:
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
        if response.status_code >= 400:
            detail = response.text[:300] if response.text else "No error details returned."
            return None, f"ElevenLabs request failed ({response.status_code}): {detail}"
        if response.content:
            return response.content, ""
        return None, "ElevenLabs returned an empty audio payload."
    except Exception as exc:
        return None, f"ElevenLabs request exception: {exc}"

    return None, "Unknown ElevenLabs audio generation error."


def _elevenlabs_single_voice_mp3_bytes(text: str) -> tuple[bytes | None, str]:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID_A")
    if not api_key or not voice_id:
        return None, "Missing ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID_A for single-voice fallback."

    base_url = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")
    output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
    model_id = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")

    # Remove basic markdown markers before one-speaker narration.
    clean_text = re.sub(r"[#*_`]", " ", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    clean_text = clean_text[:ELEVENLABS_MAX_CHARS]
    if not clean_text:
        return None, "Transcript was empty after cleanup."

    try:
        response = requests.post(
            f"{base_url}/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            params={"output_format": output_format},
            json={
                "text": clean_text[:5000],
                "model_id": model_id,
            },
            timeout=180,
        )
        if response.status_code >= 400:
            detail = response.text[:300] if response.text else "No error details returned."
            return None, f"Single-voice TTS failed ({response.status_code}): {detail}"
        if response.content:
            return response.content, ""
        return None, "Single-voice TTS returned empty audio payload."
    except Exception as exc:
        return None, f"Single-voice TTS exception: {exc}"


def _gtts_mp3_bytes(text: str) -> tuple[bytes | None, str]:
    if gTTS is None:
        return None, "gTTS fallback is not installed."

    clean_text = re.sub(r"[#*_`]", " ", text)
    clean_text = re.sub(
        r"\s+", " ", clean_text).strip()[: min(ELEVENLABS_MAX_CHARS, 3000)]
    if not clean_text:
        return None, "No text available for gTTS fallback."

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
            tts = gTTS(text=clean_text, lang="en", tld="com")
            tts.save(tmp.name)
            tmp.seek(0)
            return tmp.read(), ""
    except Exception as exc:
        return None, f"gTTS fallback failed: {exc}"


def generate_report(store: NotebookStore, username: str, notebook_id: str, extra_prompt: str = "") -> Path:
    material = _load_material(store, username, notebook_id)
    prompt = (
        "Create a polished markdown study report.\n\n"
        "Required format:\n"
        "# Title\n"
        "## Executive Summary\n"
        "- 3 to 5 bullet points\n"
        "## Key Concepts\n"
        "### Concept 1\n"
        "Paragraph explanation.\n"
        "### Concept 2\n"
        "Paragraph explanation.\n"
        "## Numbered Examples\n"
        "1. Example name\n"
        "   - Steps\n"
        "   - Why it matters\n"
        "2. Example name\n"
        "   - Steps\n"
        "   - Why it matters\n"
        "## Open Questions\n"
        "- Bullet list\n"
        "## Actionable Review Plan\n"
        "1. Step one\n"
        "2. Step two\n\n"
        "Rules: keep formatting clean, include bullets and numbered items, use short readable paragraphs, "
        "and ground claims in source material.\n\n"
        f"Additional user instruction: {extra_prompt or 'None'}\n\n"
        "Source material:\n"
        f"{material}"
    )
    text = generate_text(prompt, REPORT_SYSTEM_PROMPT,
                         fallback_text=_fallback_report(material))
    text = _format_report_markdown(text)
    return store.save_artifact_text(username, notebook_id, "report", ".md", text)


def generate_quiz(store: NotebookStore, username: str, notebook_id: str, extra_prompt: str = "") -> Path:
    material = _load_material(store, username, notebook_id)
    prompt = (
        "Create a polished markdown quiz with 8 questions.\n\n"
        "Required format:\n"
        "# Quiz Title\n"
        "## Questions\n"
        "1. Question text\n"
        "A. Choice A\n"
        "B. Choice B\n"
        "C. Choice C\n"
        "D. Choice D\n"
        "2. Question text\n"
        "A. Choice A\n"
        "B. Choice B\n"
        "C. Choice C\n"
        "D. Choice D\n"
        "(repeat)\n"
        "## Answer Key\n"
        "1. B - short explanation\n"
        "2. D - short explanation\n\n"
        "Rules: each answer choice must be on its own line; leave blank lines between questions; "
        "keep wording clear and student-friendly; ground all questions in the source material.\n\n"
        f"Additional user instruction: {extra_prompt or 'None'}\n\n"
        "Source material:\n"
        f"{material}"
    )
    text = generate_text(prompt, QUIZ_SYSTEM_PROMPT,
                         fallback_text=_fallback_quiz(material))
    text = _format_quiz_markdown(text)
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
    transcript = generate_text(
        prompt, PODCAST_SYSTEM_PROMPT, fallback_text=_fallback_podcast(material))
    transcript_path = store.save_artifact_text(
        username, notebook_id, "podcast", ".md", transcript)

    audio_path = None
    audio_error = ""
    turns = _parse_dialogue_turns(transcript)
    audio_bytes, audio_error = _elevenlabs_dialogue_mp3_bytes(turns)
    if not audio_bytes:
        single_bytes, single_error = _elevenlabs_single_voice_mp3_bytes(
            transcript)
        if single_bytes:
            audio_bytes = single_bytes
            audio_error = ""
        elif single_error:
            audio_error = f"{audio_error} | {single_error}" if audio_error else single_error

    if not audio_bytes:
        gtts_bytes, gtts_error = _gtts_mp3_bytes(transcript)
        if gtts_bytes:
            audio_bytes = gtts_bytes
            audio_error = ""
        elif gtts_error:
            audio_error = f"{audio_error} | {gtts_error}" if audio_error else gtts_error

    if audio_bytes:
        try:
            audio_path = store.save_artifact_bytes(
                username,
                notebook_id,
                "podcast",
                ".mp3",
                audio_bytes,
            )
        except Exception as exc:
            audio_path = None
            audio_error = f"Failed to save audio artifact: {exc}"

    return {
        "transcript_path": str(transcript_path),
        "audio_path": str(audio_path) if audio_path else "",
        "audio_error": audio_error,
    }


def list_artifacts(store: NotebookStore, username: str, notebook_id: str) -> list[dict[str, Any]]:
    return store.list_artifacts(username, notebook_id)
