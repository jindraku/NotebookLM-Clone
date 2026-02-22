---
title: Notebooklm Clone
emoji: 🌍
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: false
---

# NotebookLM Clone (Gradio + RAG)

A full-stack NotebookLM-style app for Hugging Face Spaces with:
- Per-user notebook isolation
- Source ingestion (`.pdf`, `.pptx`, `.txt`, web URL)
- Retrieval-augmented chat with citations
- Artifact generation (report, quiz, podcast transcript + optional mp3)
- Persistent on-disk project structure in `/data` (ephemeral on free HF tier)

## 1) Project Structure

```text
notebooklm-clone/
├── app.py
├── backend/
│   ├── artifacts.py
│   ├── chat.py
│   ├── ingest.py
│   ├── llm.py
│   └── rag.py
├── storage/
│   └── notebook_store.py
├── requirements.txt
└── docs/
    ├── ARCHITECTURE.md
    └── RAG_EXPERIMENTS.md
```

Data layout created at runtime:

```text
/data/
└── users/
    └── <username>/
        └── notebooks/
            ├── index.json
            └── <notebook-uuid>/
                ├── files_raw/
                ├── files_extracted/
                ├── chroma/
                ├── chat/messages.jsonl
                └── artifacts/
                    ├── reports/
                    ├── quizzes/
                    └── podcasts/
```

## 2) Local Setup

```bash
cd notebooklm-clone
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the URL printed by Gradio.

## 3) Environment Variables

- `OPENAI_API_KEY` (optional): enables higher quality generation for chat/artifacts.
- `OPENAI_MODEL` (optional): default is `gpt-4o-mini`.
- `DATA_ROOT` (optional): default is `data`.
- `DEMO_USER` (optional): default local username when OAuth user is unavailable.

If no `OPENAI_API_KEY` is set, the app still works with a fallback response mode.

## 4) Hugging Face Space Setup

1. Create a Gradio Space and connect this repo.
2. In Space variables/secrets set:
   - `OPENAI_API_KEY` (Secret, optional but recommended)
   - `OPENAI_MODEL` (Variable, optional)
3. Enable Hugging Face OAuth for identity-aware per-user storage.
4. Push to `main`; app boots from `app.py`.

## 5) GitHub Actions Deployment to HF Space

Workflow file: `.github/workflows/deploy-to-hf-space.yml`

Configure these in GitHub:
- Repository secret: `HF_TOKEN` (write token for Hugging Face)
- Repository variable: `HF_SPACE_ID` (example: `your-username/your-space-name`)

On push to `main`, GitHub Action mirrors `main` to your Space.

## 6) Current Feature Coverage

Implemented now:
- Notebook CRUD
- Per-user storage isolation
- Ingestion from file + URL
- Chroma vector indexing and retrieval
- Chat with citation display
- Report/quiz generation and persistence
- Podcast transcript generation + optional MP3 (via `gTTS`)

Recommended next upgrades:
- Add source enable/disable toggles per chat request
- Add advanced RAG variants and benchmarking UI
- Improve citation granularity with page/slide offsets
