# Architecture and Data Flow

## Modules

- `app.py`
  - Gradio UI and callback wiring
  - Notebook manager, ingestion controls, chat, artifact UI
  - Identity resolution through `gr.Request`

- `storage/notebook_store.py`
  - Filesystem-backed persistence
  - Per-user notebook CRUD
  - Chat history append/read (`jsonl`)
  - Artifact save/list helpers

- `backend/ingest.py`
  - Extract text from PDF/PPTX/TXT and URLs
  - Chunking
  - Embedding and indexing into ChromaDB

- `backend/rag.py`
  - Similarity retrieval from notebook-specific Chroma collection
  - Returns chunks with metadata and distance score

- `backend/chat.py`
  - Retrieves top-k context
  - Prompts LLM and returns answer
  - Persists user/assistant messages with citations

- `backend/artifacts.py`
  - Generates report/quiz/podcast transcript from notebook corpus
  - Persists outputs under notebook artifact folders
  - Optional podcast MP3 synthesis

- `backend/llm.py`
  - Unified generation wrapper
  - Uses OpenAI if `OPENAI_API_KEY` is present
  - Otherwise returns deterministic fallback output

## End-to-End Flows

### Ingestion flow
1. User selects notebook.
2. Upload file or enter URL.
3. Text extraction + chunking.
4. Chunk embeddings saved to notebook Chroma DB.
5. Extracted text saved for traceability and artifact generation.

### Chat flow
1. User asks question.
2. Retrieve top-k chunks from notebook Chroma DB.
3. Build grounded prompt with source metadata.
4. Generate answer.
5. Save chat turn and citations to `messages.jsonl`.

### Artifact flow
1. Read extracted source text for notebook.
2. Generate markdown artifact from corpus (+ optional user prompt).
3. Save to notebook artifact directory.
4. Optional podcast MP3 generation from transcript.

## Data Isolation

All notebook, chat, and artifact operations are rooted under:

`/data/users/<username>/notebooks/<notebook-id>/...`

The username comes from Hugging Face OAuth (`request.username`) when available.
