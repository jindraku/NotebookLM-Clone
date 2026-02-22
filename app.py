from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr

from backend.artifacts import generate_podcast, generate_quiz, generate_report, list_artifacts
from backend.chat import chat_with_notebook
from backend.ingest import ingest_many_files, ingest_url
from storage.notebook_store import NotebookStore


DATA_ROOT = os.getenv("DATA_ROOT", "data")
store = NotebookStore(DATA_ROOT)
RUNNING_ON_SPACE = bool(os.getenv("SPACE_ID"))


def current_username(request: gr.Request | None) -> str:
    if request is not None:
        # HF Spaces OAuth populates request.username when enabled.
        username = getattr(request, "username", None)
        if username:
            return str(username)
        headers = getattr(request, "headers", {}) or {}
        user_header = headers.get("x-forwarded-user") if isinstance(headers, dict) else None
        if user_header:
            return str(user_header)
    return os.getenv("DEMO_USER", "local-user")


def notebook_choices(username: str) -> tuple[list[str], str | None, str]:
    notebooks = store.list_notebooks(username)
    ids = [n["id"] for n in notebooks]
    selected = ids[0] if ids else None
    lines = [f"- `{n['id']}`: {n['name']}" for n in notebooks]
    summary = "\n".join(lines) if lines else "No notebooks yet."
    return ids, selected, summary


def refresh_notebooks(request: gr.Request | None):
    username = current_username(request)
    choices, selected, summary = notebook_choices(username)
    return gr.update(choices=choices, value=selected), summary


def create_notebook(name: str, request: gr.Request | None):
    username = current_username(request)
    notebook = store.create_notebook(username, name)
    choices, selected, summary = notebook_choices(username)
    return gr.update(choices=choices, value=notebook["id"]), summary, f"Created notebook `{notebook['name']}`."


def rename_notebook(notebook_id: str, new_name: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return gr.update(), "Select a notebook first.", "No notebook selected."
    ok = store.rename_notebook(username, notebook_id, new_name)
    choices, selected, summary = notebook_choices(username)
    status = "Renamed notebook." if ok else "Notebook rename failed."
    return gr.update(choices=choices, value=selected or notebook_id), summary, status


def delete_notebook(notebook_id: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return gr.update(), "No notebook selected.", []
    ok = store.delete_notebook(username, notebook_id)
    choices, selected, summary = notebook_choices(username)
    chat_state = load_chat(selected, request) if selected else []
    status = "Deleted notebook." if ok else "Notebook not found."
    return gr.update(choices=choices, value=selected), f"{status}\n\n{summary}", chat_state


def load_chat(notebook_id: str | None, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return []
    rows = store.load_messages(username, notebook_id)
    pairs: list[list[str]] = []
    pending_user = ""
    for row in rows:
        role = row.get("role", "")
        content = row.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant":
            pairs.append([pending_user, content])
            pending_user = ""
    return pairs


def ingest_files_callback(notebook_id: str, files: list[Any] | None, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return "Select a notebook first."
    if not files:
        return "Choose one or more files first."

    results = ingest_many_files(store, username, notebook_id, files)
    lines = []
    for r in results:
        lines.append(f"- {r.source_name}: {r.status} ({r.num_chunks} chunks) {r.detail}")
    return "\n".join(lines)


def ingest_url_callback(notebook_id: str, url: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return "Select a notebook first."
    if not url.strip():
        return "Enter a URL first."

    result = ingest_url(store, username, notebook_id, url.strip())
    return f"{result.source_name}: {result.status} ({result.num_chunks} chunks) {result.detail}"


def chat_callback(notebook_id: str, message: str, history: list[list[str]] | None, request: gr.Request | None):
    username = current_username(request)
    history = history or []
    if not notebook_id:
        return history, "Select a notebook first.", ""
    if not message.strip():
        return history, "", ""

    result = chat_with_notebook(store, username, notebook_id, message.strip())
    new_history = history + [[message.strip(), result["answer"]]]

    citations = result.get("citations", [])
    if citations:
        citation_text = "\n".join(
            [
                f"- `{c['source_name']}` chunk {c['chunk_index']} (distance={c['score']})"
                for c in citations
            ]
        )
    else:
        citation_text = "No citations returned."
    return new_history, "", citation_text


def _artifact_options(username: str, notebook_id: str) -> tuple[list[str], str | None, str]:
    rows = list_artifacts(store, username, notebook_id)
    paths = [r["path"] for r in rows]
    summary = "\n".join([f"- {r['type']}: {r['name']}" for r in rows]) if rows else "No artifacts yet."
    selected = paths[0] if paths else None
    return paths, selected, summary


def refresh_artifacts(notebook_id: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return gr.update(choices=[], value=None), "Select a notebook first."
    choices, selected, summary = _artifact_options(username, notebook_id)
    return gr.update(choices=choices, value=selected), summary


def generate_report_callback(notebook_id: str, prompt: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return gr.update(), "Select a notebook first.", None
    path = generate_report(store, username, notebook_id, prompt)
    choices, selected, summary = _artifact_options(username, notebook_id)
    return gr.update(choices=choices, value=str(path)), f"Generated report: `{path.name}`\n\n{summary}", str(path)


def generate_quiz_callback(notebook_id: str, prompt: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return gr.update(), "Select a notebook first.", None
    path = generate_quiz(store, username, notebook_id, prompt)
    choices, selected, summary = _artifact_options(username, notebook_id)
    return gr.update(choices=choices, value=str(path)), f"Generated quiz: `{path.name}`\n\n{summary}", str(path)


def generate_podcast_callback(notebook_id: str, prompt: str, request: gr.Request | None):
    username = current_username(request)
    if not notebook_id:
        return gr.update(), "Select a notebook first.", None, None
    result = generate_podcast(store, username, notebook_id, prompt)
    choices, selected, summary = _artifact_options(username, notebook_id)
    transcript = result.get("transcript_path")
    audio = result.get("audio_path")
    status = [f"Generated podcast transcript: `{Path(transcript).name}`" if transcript else "Transcript generation failed."]
    if audio:
        status.append(f"Generated audio: `{Path(audio).name}`")
    else:
        status.append("Audio generation unavailable in this runtime.")
    status.append("")
    status.append(summary)
    return gr.update(choices=choices, value=audio or transcript), "\n".join(status), transcript, audio


def select_artifact(path: str):
    if not path:
        return None, None
    lower = path.lower()
    if lower.endswith(".mp3"):
        return None, path
    return path, None


with gr.Blocks(title="NotebookLM Clone") as demo:
    gr.Markdown("# NotebookLM Clone\nPer-user notebooks, RAG chat with citations, and artifact generation.")

    if RUNNING_ON_SPACE and hasattr(gr, "LoginButton"):
        gr.LoginButton(value="Sign in with Hugging Face")

    notebook_status = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Notebook Manager")
            notebook_dropdown = gr.Dropdown(choices=[], value=None, label="Notebook ID")
            notebook_summary = gr.Markdown("No notebooks yet.")
            new_notebook_name = gr.Textbox(label="New Notebook Name", placeholder="AI Lecture Notes")
            create_btn = gr.Button("Create")
            rename_name = gr.Textbox(label="Rename Notebook", placeholder="New notebook name")
            rename_btn = gr.Button("Rename")
            delete_btn = gr.Button("Delete Notebook")

            gr.Markdown("## Ingest Sources")
            file_input = gr.File(file_count="multiple", label="Upload files (.pdf, .pptx, .txt)")
            ingest_files_btn = gr.Button("Ingest Files")
            url_input = gr.Textbox(label="Web URL", placeholder="https://example.com/article")
            ingest_url_btn = gr.Button("Ingest URL")
            ingest_status = gr.Markdown("")

            gr.Markdown("## Artifacts")
            artifact_prompt = gr.Textbox(
                label="Artifact Prompt (optional)",
                placeholder="Focus on topic X and compare against topic Y.",
            )
            with gr.Row():
                gen_report_btn = gr.Button("Generate Report")
                gen_quiz_btn = gr.Button("Generate Quiz")
                gen_podcast_btn = gr.Button("Generate Podcast")

            artifact_dropdown = gr.Dropdown(choices=[], value=None, label="Saved Artifact Path")
            artifact_status = gr.Markdown("No artifacts yet.")
            artifact_md_file = gr.File(label="Download Markdown Artifact")
            artifact_audio = gr.Audio(label="Podcast Audio", type="filepath")

        with gr.Column(scale=2):
            gr.Markdown("## Chat")
            chatbot = gr.Chatbot(height=520)
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Ask questions about your sources...", scale=5)
                send_btn = gr.Button("Send", scale=1)
            citations = gr.Markdown("Citations appear here after each assistant response.")

    demo.load(refresh_notebooks, inputs=None, outputs=[notebook_dropdown, notebook_summary])

    create_btn.click(
        create_notebook,
        inputs=[new_notebook_name],
        outputs=[notebook_dropdown, notebook_summary, notebook_status],
    )
    rename_btn.click(
        rename_notebook,
        inputs=[notebook_dropdown, rename_name],
        outputs=[notebook_dropdown, notebook_summary, notebook_status],
    )
    delete_btn.click(
        delete_notebook,
        inputs=[notebook_dropdown],
        outputs=[notebook_dropdown, notebook_summary, chatbot],
    )

    notebook_dropdown.change(load_chat, inputs=[notebook_dropdown], outputs=[chatbot]).then(
        refresh_artifacts,
        inputs=[notebook_dropdown],
        outputs=[artifact_dropdown, artifact_status],
    )

    ingest_files_btn.click(
        ingest_files_callback,
        inputs=[notebook_dropdown, file_input],
        outputs=[ingest_status],
    )
    ingest_url_btn.click(
        ingest_url_callback,
        inputs=[notebook_dropdown, url_input],
        outputs=[ingest_status],
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
        outputs=[artifact_dropdown, artifact_status, artifact_md_file],
    )
    gen_quiz_btn.click(
        generate_quiz_callback,
        inputs=[notebook_dropdown, artifact_prompt],
        outputs=[artifact_dropdown, artifact_status, artifact_md_file],
    )
    gen_podcast_btn.click(
        generate_podcast_callback,
        inputs=[notebook_dropdown, artifact_prompt],
        outputs=[artifact_dropdown, artifact_status, artifact_md_file, artifact_audio],
    )

    artifact_dropdown.change(select_artifact, inputs=[artifact_dropdown], outputs=[artifact_md_file, artifact_audio])


if __name__ == "__main__":
    demo.launch()
