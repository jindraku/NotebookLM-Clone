from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import gradio as gr

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from backend.artifacts import generate_podcast, generate_quiz, generate_report, list_artifacts
from backend.chat import chat_with_notebook
from backend.ingest import ingest_many_files, ingest_url
from storage.notebook_store import NotebookStore

if load_dotenv is not None:
    # Local convenience: load .env automatically so API keys are available without shell export.
    load_dotenv(override=False)


DATA_ROOT = os.getenv("DATA_ROOT", "data")
store = NotebookStore(DATA_ROOT)
RUNNING_ON_SPACE = bool(os.getenv("SPACE_ID"))
OAUTH_ENABLED = bool(os.getenv("OAUTH_CLIENT_ID"))

APP_CSS = """
:root {
  --bg-main: #050a18;
  --bg-panel: #0b1429;
  --bg-soft: #1a2740;
  --text-main: #edf2ff;
  --text-dim: #a9b4ce;
  --accent: #5f54f5;
  --accent-soft: #2b3d71;
  --border: #263455;
}

.gradio-container {
  background: radial-gradient(circle at 20% 0%, #101d3a 0%, var(--bg-main) 45%, #040816 100%);
  color: var(--text-main);
}

#app-shell {
  max-width: 1500px;
  margin: 0 auto;
}

.app-title h1 {
  margin: 0;
  font-size: 2.25rem;
}

.card {
  background: linear-gradient(160deg, #101d38 0%, #0a1224 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px;
}

.left-section-title {
  color: var(--text-main);
  font-weight: 700;
  margin: 10px 0 6px 0;
}

.left-separator {
  border-top: 1px solid var(--border);
  margin: 14px 0;
}

.status-pill {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 8px 12px;
  background: #0a1224;
}

#main-tabs .tab-nav {
  border-bottom: 1px solid var(--border) !important;
}

#main-tabs .tab-nav button,
#artifact-tabs .tab-nav button {
  color: var(--text-dim) !important;
  font-weight: 700 !important;
}

#main-tabs .tab-nav button.selected,
#artifact-tabs .tab-nav button.selected {
  color: #d9ddff !important;
  border-bottom-color: var(--accent) !important;
}

.primary-btn button {
  background: linear-gradient(90deg, #5c55f3 0%, #4c44d8 100%) !important;
  border: 1px solid #6f68ff !important;
  color: #fff !important;
  font-weight: 700 !important;
}

.secondary-btn button {
  background: #1c2740 !important;
  border: 1px solid var(--border) !important;
  color: var(--text-main) !important;
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
  margin-bottom: 8px;
}
"""


def current_username(request: gr.Request | None) -> str:
    if request is not None:
        username = getattr(request, "username", None)
        if username:
            return str(username)

        headers = getattr(request, "headers", {}) or {}
        if isinstance(headers, dict):
            lowered = {str(k).lower(): v for k, v in headers.items()}
            for key in [
                "x-forwarded-user",
                "x-forwarded-preferred-username",
                "x-hf-username",
                "x-hf-user",
                "x-user",
                "x-auth-request-user",
            ]:
                value = lowered.get(key)
                if value:
                    return str(value)

            # Some proxies may pass serialized auth context.
            for key in ["x-auth-request", "x-userinfo", "x-hf-userinfo"]:
                raw = lowered.get(key)
                if not raw:
                    continue
                try:
                    parsed = json.loads(str(raw))
                    for name_key in ["preferred_username", "username", "name", "sub"]:
                        if parsed.get(name_key):
                            return str(parsed[name_key])
                except Exception:
                    continue
    return os.getenv("DEMO_USER", "local-user")


def user_badge_text(request: gr.Request | None) -> str:
    return f"Logged in as: **{current_username(request)}**"


def notebook_choices(username: str) -> tuple[list[str], str | None, str]:
    notebooks = store.list_notebooks(username)
    ids = [n["id"] for n in notebooks]
    selected = ids[0] if ids else None
    lines = [f"- `{n['id']}`" for n in notebooks]
    summary = "\n".join(lines) if lines else "No notebooks yet."
    return ids, selected, summary


def source_list_markdown(username: str, notebook_id: str | None) -> str:
    if not notebook_id:
        return "_No notebook selected._"
    paths = store.notebook_paths(username, notebook_id)
    if not paths.files_raw.exists():
        return "_No sources ingested._"

    files = sorted([p.name for p in paths.files_raw.iterdir() if p.is_file()])
    if not files:
        return "_No sources ingested._"

    return "\n".join([f"- {name}" for name in files])


def load_chat(notebook_id: str | None, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return []
    rows = store.load_messages(username, notebook_id)
    messages: list[dict[str, str]] = []
    for row in rows:
        role = row.get("role", "")
        content = row.get("content", "")
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": content})
    return messages


def _artifact_rows(username: str, notebook_id: str | None) -> list[dict[str, str]]:
    if not notebook_id:
        return []
    rows = list_artifacts(store, username, notebook_id)
    def _mtime(row: dict[str, str]) -> float:
        try:
            return Path(row["path"]).stat().st_mtime
        except Exception:
            return 0.0

    return sorted(rows, key=_mtime, reverse=True)


def _artifact_file_markdown(rows: list[dict[str, str]], kind: str) -> str:
    files = [r["name"] for r in rows if r["type"] == kind]
    if not files:
        return "_No files yet._"
    return "\n".join([f"- {name}" for name in files])


def _first_report_preview(rows: list[dict[str, str]]) -> str:
    report_paths = [r["path"] for r in rows if r["type"] == "report" and r["path"].endswith(".md")]
    if not report_paths:
        return "## Report Preview\n\nGenerate a report to see it here."

    path = Path(report_paths[0])
    if not path.exists():
        return "## Report Preview\n\nReport file not found."

    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:12000] if text else "## Report Preview\n\nReport is empty."


def _first_quiz_preview(rows: list[dict[str, str]]) -> str:
    quiz_paths = [r["path"] for r in rows if r["type"] == "quiz" and r["path"].endswith(".md")]
    if not quiz_paths:
        return "## Quiz Preview\n\nGenerate a quiz to see it here."

    path = Path(quiz_paths[0])
    if not path.exists():
        return "## Quiz Preview\n\nQuiz file not found."

    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:12000] if text else "## Quiz Preview\n\nQuiz is empty."


def _artifact_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No artifacts yet."
    return "\n".join([f"- {r['type']}: {r['name']}" for r in rows])


def refresh_artifact_panel(notebook_id: str | None, request: gr.Request | None):
    username = current_username(request)
    rows = _artifact_rows(username, notebook_id)
    choices = [(f'{r["type"]}: {r["name"]}', r["path"]) for r in rows]
    selected = choices[0][1] if choices else None
    report_files = _artifact_file_markdown(rows, "report")
    quiz_files = _artifact_file_markdown(rows, "quiz")
    podcast_files = _artifact_file_markdown(rows, "podcast")
    report_preview = _first_report_preview(rows)
    quiz_preview = _first_quiz_preview(rows)

    md_file = selected if selected and selected.lower().endswith(".md") else None
    audio_file = selected if selected and selected.lower().endswith(".mp3") else None

    return (
        gr.update(choices=choices, value=selected),
        gr.update(),
        report_files,
        quiz_files,
        podcast_files,
        report_preview,
        quiz_preview,
        md_file,
        audio_file,
    )


def refresh_artifact_panel_keep_status(notebook_id: str | None, request: gr.Request | None):
    dropdown, _status, report_files, quiz_files, podcast_files, report_preview, quiz_preview, md_file, audio_file = (
        refresh_artifact_panel(notebook_id, request)
    )
    return (
        dropdown,
        report_files,
        quiz_files,
        podcast_files,
        report_preview,
        quiz_preview,
        md_file,
        audio_file,
    )


def refresh_notebooks(request: gr.Request | None):
    username = current_username(request)
    choices, selected, summary = notebook_choices(username)
    source_md = source_list_markdown(username, selected)
    return gr.update(choices=choices, value=selected), summary, source_md, user_badge_text(request)


def create_notebook(name: str, request: gr.Request | None):
    username = current_username(request)
    notebook = store.create_notebook(username, name)
    choices, selected, summary = notebook_choices(username)
    status = f"Created notebook `{notebook['id']}`."
    source_md = source_list_markdown(username, notebook["id"])
    return gr.update(choices=choices, value=notebook["id"]), summary, status, source_md


def rename_notebook(notebook_id: str, new_name: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return gr.update(), "No notebook selected.", "No notebook selected."
    ok = store.rename_notebook(username, notebook_id, new_name)
    choices, selected, summary = notebook_choices(username)
    return gr.update(choices=choices, value=selected or notebook_id), summary, (
        "Notebook renamed." if ok else "Notebook rename failed."
    )


def delete_notebook(notebook_id: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return gr.update(), "No notebook selected.", "No notebook selected.", []

    ok = store.delete_notebook(username, notebook_id)
    choices, selected, summary = notebook_choices(username)
    status = "Deleted notebook." if ok else "Notebook not found."
    chat_state = load_chat(selected, request) if selected else []
    return gr.update(choices=choices, value=selected), summary, status, chat_state


def ingest_files_callback(notebook_id: str, files: list[Any] | None, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return "Select a notebook first.", source_list_markdown(username, notebook_id)
    if not files:
        return "Choose one or more files first.", source_list_markdown(username, notebook_id)

    results = ingest_many_files(store, username, notebook_id, files)
    lines = [f"- {r.source_name}: {r.status} ({r.num_chunks} chunks)" for r in results]
    return "\n".join(lines), source_list_markdown(username, notebook_id)


def ingest_url_callback(notebook_id: str, url: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return "Select a notebook first.", source_list_markdown(username, notebook_id)
    if not url.strip():
        return "Enter a URL first.", source_list_markdown(username, notebook_id)

    result = ingest_url(store, username, notebook_id, url.strip())
    status = f"{result.source_name}: {result.status} ({result.num_chunks} chunks)"
    return status, source_list_markdown(username, notebook_id)


def refresh_sources_for_selected(notebook_id: str | None, request: gr.Request | None) -> str:
    return source_list_markdown(current_username(request), notebook_id)


def chat_callback(notebook_id: str, message: str, history: list[Any] | None, request: gr.Request | None):
    username = current_username(request)
    history = history or []
    if not notebook_id:
        return history, "Select a notebook first.", ""
    if not message.strip():
        return history, "", ""

    result = chat_with_notebook(store, username, notebook_id, message.strip())
    new_history = history + [
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": result["answer"]},
    ]

    citations = result.get("citations", [])
    if citations:
        citation_text = "\n".join(
            [f"- `{c['source_name']}` chunk {c['chunk_index']} (distance={c['score']})" for c in citations]
        )
    else:
        citation_text = "No citations returned."
    return new_history, "", citation_text


def generate_report_callback(notebook_id: str, prompt: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return "Select a notebook first."
    path = generate_report(store, username, notebook_id, prompt)
    return f"Report generated: `{path.name}`"


def generate_quiz_callback(notebook_id: str, prompt: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return "Select a notebook first."
    path = generate_quiz(store, username, notebook_id, prompt)
    return f"Quiz generated: `{path.name}`"


def generate_podcast_callback(notebook_id: str, prompt: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return "Select a notebook first."
    result = generate_podcast(store, username, notebook_id, prompt)
    transcript = result.get("transcript_path")
    audio = result.get("audio_path")
    audio_error = result.get("audio_error", "")
    if transcript and audio:
        return f"Podcast generated: `{Path(transcript).name}` + `{Path(audio).name}`"
    if transcript:
        detail = f" Reason: {audio_error}" if audio_error else ""
        return f"Podcast transcript generated: `{Path(transcript).name}` (audio unavailable).{detail}"
    return "Podcast generation failed."


def select_artifact(path: str, notebook_id: str | None, request: gr.Request | None):
    username = current_username(request)
    md_file = path if path and path.lower().endswith(".md") else None
    audio_file = path if path and path.lower().endswith(".mp3") else None
    rows = _artifact_rows(username, notebook_id)
    report_preview = _first_report_preview(rows)
    quiz_preview = _first_quiz_preview(rows)

    if md_file and Path(md_file).exists():
        preview_text = Path(md_file).read_text(encoding="utf-8", errors="ignore")[:12000]
        lower = md_file.lower()
        if "/reports/" in lower:
            report_preview = preview_text
        elif "/quizzes/" in lower:
            quiz_preview = preview_text

    return md_file, audio_file, report_preview, quiz_preview


with gr.Blocks(title="NotebookLM Clone", fill_height=True, elem_id="app-shell") as demo:
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=340):
            gr.Markdown("# NotebookLM Clone", elem_classes=["app-title"])
            gr.Markdown("## Notebooks", elem_classes=["left-section-title"])

            with gr.Column(elem_classes=["card"]):
                notebook_dropdown = gr.Dropdown(choices=[], value=None, label="Select Notebook")

            new_notebook_name = gr.Textbox(label="", placeholder="new notebook name")
            create_btn = gr.Button("+ New", elem_classes=["primary-btn"])

            with gr.Accordion("Manage Notebook", open=False):
                rename_name = gr.Textbox(label="Rename", placeholder="new name")
                rename_btn = gr.Button("Rename", elem_classes=["secondary-btn"])
                delete_btn = gr.Button("Delete", elem_classes=["secondary-btn"])

            gr.Markdown("<div class='left-separator'></div>")
            gr.Markdown("### Ingested Sources")
            source_list = gr.Markdown("_No sources ingested._", elem_classes=["card"])

            notebook_summary = gr.Markdown("No notebooks yet.")
            notebook_status = gr.Markdown("", elem_classes=["status-pill"])

        with gr.Column(scale=3, min_width=760):
            with gr.Row(elem_classes=["topbar"]):
                user_badge = gr.Markdown("Logged in as: **local-user**")
                if RUNNING_ON_SPACE and OAUTH_ENABLED and hasattr(gr, "LoginButton"):
                    gr.LoginButton(value="Sign in with Hugging Face")

            with gr.Tabs(elem_id="main-tabs"):
                with gr.Tab("Sources"):
                    with gr.Column(elem_classes=["card"]):
                        file_input = gr.File(
                            file_count="multiple",
                            label="Upload files (.pdf, .pptx, .txt, .csv, .xlsx)",
                        )
                        ingest_files_btn = gr.Button("Ingest Files", elem_classes=["primary-btn"])
                        url_input = gr.Textbox(label="Web URL", placeholder="https://example.com/article")
                        ingest_url_btn = gr.Button("Ingest URL", elem_classes=["secondary-btn"])
                        ingest_status = gr.Markdown("")

                with gr.Tab("Chat"):
                    chatbot = gr.Chatbot(height=520)
                    with gr.Row():
                        chat_input = gr.Textbox(placeholder="Ask questions about your sources...", scale=6)
                        send_btn = gr.Button("Send", scale=1, elem_classes=["primary-btn"])
                    gr.Markdown(
                        "**Try questions like:**\n"
                        "- What are the top 3 takeaways from these sources?\n"
                        "- Explain this topic like I'm new to it, with one real example.\n"
                        "- Compare source A and source B on their main argument.\n"
                        "- What evidence supports the claim about X?\n"
                        "- What important details are missing from these notes?"
                    )
                    gr.Examples(
                        examples=[
                            "What are the top 3 takeaways from these sources?",
                            "Explain this topic like I'm new to it, with one real example.",
                            "Compare the main ideas across my uploaded sources.",
                            "What evidence from the sources supports the key conclusion?",
                            "Summarize this as a short study guide with action items.",
                        ],
                        inputs=[chat_input],
                    )
                    citations = gr.Markdown("Citations appear here after each response.")

                with gr.Tab("Artifacts"):
                    artifact_prompt = gr.Textbox(
                        label="Artifact Prompt (optional)",
                        placeholder="focus on topic X and compare to topic Y",
                    )

                    artifact_dropdown = gr.Dropdown(choices=[], value=None, label="Artifact Files")
                    artifact_status = gr.Markdown("No artifacts yet.", elem_classes=["status-pill"])

                    with gr.Tabs(elem_id="artifact-tabs"):
                        with gr.Tab("Reports"):
                            gen_report_btn = gr.Button("Generate Report", elem_classes=["primary-btn"])
                            report_files = gr.Markdown("_No files yet._", elem_classes=["card"])
                            report_preview = gr.Markdown("## Report Preview\n\nGenerate a report to see it here.")

                        with gr.Tab("Quizzes"):
                            gen_quiz_btn = gr.Button("Generate Quiz", elem_classes=["primary-btn"])
                            quiz_files = gr.Markdown("_No files yet._", elem_classes=["card"])
                            quiz_preview = gr.Markdown("## Quiz Preview\n\nGenerate a quiz to see it here.")

                        with gr.Tab("Podcasts"):
                            gen_podcast_btn = gr.Button("Generate Podcast", elem_classes=["primary-btn"])
                            podcast_files = gr.Markdown("_No files yet._", elem_classes=["card"])
                            artifact_audio = gr.Audio(label="Podcast Audio", type="filepath")

                    artifact_md_file = gr.File(label="Download Markdown Artifact")

    demo.load(
        refresh_notebooks,
        inputs=None,
        outputs=[notebook_dropdown, notebook_summary, source_list, user_badge],
    ).then(
        load_chat,
        inputs=[notebook_dropdown],
        outputs=[chatbot],
    ).then(
        refresh_artifact_panel,
        inputs=[notebook_dropdown],
        outputs=[
            artifact_dropdown,
            artifact_status,
            report_files,
            quiz_files,
            podcast_files,
            report_preview,
            quiz_preview,
            artifact_md_file,
            artifact_audio,
        ],
    )

    create_btn.click(
        create_notebook,
        inputs=[new_notebook_name],
        outputs=[notebook_dropdown, notebook_summary, notebook_status, source_list],
    ).then(
        load_chat,
        inputs=[notebook_dropdown],
        outputs=[chatbot],
    ).then(
        refresh_artifact_panel,
        inputs=[notebook_dropdown],
        outputs=[
            artifact_dropdown,
            artifact_status,
            report_files,
            quiz_files,
            podcast_files,
            report_preview,
            quiz_preview,
            artifact_md_file,
            artifact_audio,
        ],
    )

    rename_btn.click(
        rename_notebook,
        inputs=[notebook_dropdown, rename_name],
        outputs=[notebook_dropdown, notebook_summary, notebook_status],
    )

    delete_btn.click(
        delete_notebook,
        inputs=[notebook_dropdown],
        outputs=[notebook_dropdown, notebook_summary, notebook_status, chatbot],
    ).then(
        refresh_artifact_panel,
        inputs=[notebook_dropdown],
        outputs=[
            artifact_dropdown,
            artifact_status,
            report_files,
            quiz_files,
            podcast_files,
            report_preview,
            quiz_preview,
            artifact_md_file,
            artifact_audio,
        ],
    ).then(
        refresh_sources_for_selected,
        inputs=[notebook_dropdown],
        outputs=[source_list],
    )

    notebook_dropdown.change(
        load_chat,
        inputs=[notebook_dropdown],
        outputs=[chatbot],
    ).then(
        refresh_artifact_panel,
        inputs=[notebook_dropdown],
        outputs=[
            artifact_dropdown,
            artifact_status,
            report_files,
            quiz_files,
            podcast_files,
            report_preview,
            quiz_preview,
            artifact_md_file,
            artifact_audio,
        ],
    ).then(
        refresh_sources_for_selected,
        inputs=[notebook_dropdown],
        outputs=[source_list],
    )

    ingest_files_btn.click(
        ingest_files_callback,
        inputs=[notebook_dropdown, file_input],
        outputs=[ingest_status, source_list],
    )

    ingest_url_btn.click(
        ingest_url_callback,
        inputs=[notebook_dropdown, url_input],
        outputs=[ingest_status, source_list],
    )

    send_btn.click(
        chat_callback,
        inputs=[notebook_dropdown, chat_input, chatbot],
        outputs=[chatbot, chat_input, citations],
    )

    chat_input.submit(
        chat_callback,
        inputs=[notebook_dropdown, chat_input, chatbot],
        outputs=[chatbot, chat_input, citations],
    )

    gen_report_btn.click(
        generate_report_callback,
        inputs=[notebook_dropdown, artifact_prompt],
        outputs=[artifact_status],
    ).then(
        refresh_artifact_panel_keep_status,
        inputs=[notebook_dropdown],
        outputs=[
            artifact_dropdown,
            report_files,
            quiz_files,
            podcast_files,
            report_preview,
            quiz_preview,
            artifact_md_file,
            artifact_audio,
        ],
    )

    gen_quiz_btn.click(
        generate_quiz_callback,
        inputs=[notebook_dropdown, artifact_prompt],
        outputs=[artifact_status],
    ).then(
        refresh_artifact_panel_keep_status,
        inputs=[notebook_dropdown],
        outputs=[
            artifact_dropdown,
            report_files,
            quiz_files,
            podcast_files,
            report_preview,
            quiz_preview,
            artifact_md_file,
            artifact_audio,
        ],
    )

    gen_podcast_btn.click(
        generate_podcast_callback,
        inputs=[notebook_dropdown, artifact_prompt],
        outputs=[artifact_status],
    ).then(
        refresh_artifact_panel_keep_status,
        inputs=[notebook_dropdown],
        outputs=[
            artifact_dropdown,
            report_files,
            quiz_files,
            podcast_files,
            report_preview,
            quiz_preview,
            artifact_md_file,
            artifact_audio,
        ],
    )

    artifact_dropdown.change(
        select_artifact,
        inputs=[artifact_dropdown, notebook_dropdown],
        outputs=[artifact_md_file, artifact_audio, report_preview, quiz_preview],
    )


if __name__ == "__main__":
    demo.launch(css=APP_CSS, ssr_mode=False)
