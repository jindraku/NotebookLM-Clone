from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_username(username: str) -> str:
    normalized = username.strip().lower()
    if not normalized:
        normalized = "anonymous"
    return re.sub(r"[^a-z0-9._-]", "_", normalized)


@dataclass
class NotebookPaths:
    root: Path
    files_raw: Path
    files_extracted: Path
    chroma: Path
    chat: Path
    artifacts: Path
    reports: Path
    quizzes: Path
    podcasts: Path


class NotebookStore:
    def __init__(self, data_root: str | Path = "data") -> None:
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def _user_root(self, username: str) -> Path:
        return self.data_root / "users" / sanitize_username(username)

    def _notebooks_root(self, username: str) -> Path:
        return self._user_root(username) / "notebooks"

    def _index_path(self, username: str) -> Path:
        return self._notebooks_root(username) / "index.json"

    def _read_index(self, username: str) -> list[dict[str, Any]]:
        index_path = self._index_path(username)
        if not index_path.exists():
            return []
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def _write_index(self, username: str, rows: list[dict[str, Any]]) -> None:
        notebooks_root = self._notebooks_root(username)
        notebooks_root.mkdir(parents=True, exist_ok=True)
        with self._index_path(username).open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    def list_notebooks(self, username: str) -> list[dict[str, Any]]:
        return sorted(self._read_index(username), key=lambda x: x.get("updated_at", ""), reverse=True)

    def create_notebook(self, username: str, name: str) -> dict[str, Any]:
        notebook_id = str(uuid4())
        now = utc_now_iso()
        row = {
            "id": notebook_id,
            "name": name.strip() or "Untitled Notebook",
            "created_at": now,
            "updated_at": now,
        }
        rows = self._read_index(username)
        rows.append(row)
        self._write_index(username, rows)

        paths = self.notebook_paths(username, notebook_id)
        for p in [
            paths.files_raw,
            paths.files_extracted,
            paths.chroma,
            paths.chat,
            paths.reports,
            paths.quizzes,
            paths.podcasts,
        ]:
            p.mkdir(parents=True, exist_ok=True)

        messages_path = paths.chat / "messages.jsonl"
        if not messages_path.exists():
            messages_path.touch()

        return row

    def rename_notebook(self, username: str, notebook_id: str, new_name: str) -> bool:
        rows = self._read_index(username)
        changed = False
        for row in rows:
            if row.get("id") == notebook_id:
                row["name"] = new_name.strip() or row.get("name", "Untitled Notebook")
                row["updated_at"] = utc_now_iso()
                changed = True
                break
        if changed:
            self._write_index(username, rows)
        return changed

    def delete_notebook(self, username: str, notebook_id: str) -> bool:
        rows = self._read_index(username)
        filtered = [r for r in rows if r.get("id") != notebook_id]
        if len(filtered) == len(rows):
            return False
        self._write_index(username, filtered)

        n_root = self._notebooks_root(username) / notebook_id
        if n_root.exists():
            shutil.rmtree(n_root)
        return True

    def notebook_paths(self, username: str, notebook_id: str) -> NotebookPaths:
        root = self._notebooks_root(username) / notebook_id
        artifacts = root / "artifacts"
        return NotebookPaths(
            root=root,
            files_raw=root / "files_raw",
            files_extracted=root / "files_extracted",
            chroma=root / "chroma",
            chat=root / "chat",
            artifacts=artifacts,
            reports=artifacts / "reports",
            quizzes=artifacts / "quizzes",
            podcasts=artifacts / "podcasts",
        )

    def touch_notebook(self, username: str, notebook_id: str) -> None:
        rows = self._read_index(username)
        for row in rows:
            if row.get("id") == notebook_id:
                row["updated_at"] = utc_now_iso()
                break
        self._write_index(username, rows)

    def save_message(self, username: str, notebook_id: str, role: str, content: str, citations: list[dict[str, Any]] | None = None) -> None:
        paths = self.notebook_paths(username, notebook_id)
        paths.chat.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": utc_now_iso(),
            "role": role,
            "content": content,
            "citations": citations or [],
        }
        with (paths.chat / "messages.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.touch_notebook(username, notebook_id)

    def load_messages(self, username: str, notebook_id: str) -> list[dict[str, Any]]:
        path = self.notebook_paths(username, notebook_id).chat / "messages.jsonl"
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def save_artifact_text(
        self,
        username: str,
        notebook_id: str,
        artifact_type: str,
        extension: str,
        content: str,
    ) -> Path:
        artifact_dir = self._artifact_dir(username, notebook_id, artifact_type)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / self._next_artifact_name(artifact_dir, artifact_type, extension)
        path.write_text(content, encoding="utf-8")
        self.touch_notebook(username, notebook_id)
        return path

    def save_artifact_bytes(
        self,
        username: str,
        notebook_id: str,
        artifact_type: str,
        extension: str,
        content: bytes,
    ) -> Path:
        artifact_dir = self._artifact_dir(username, notebook_id, artifact_type)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / self._next_artifact_name(artifact_dir, artifact_type, extension)
        path.write_bytes(content)
        self.touch_notebook(username, notebook_id)
        return path

    def list_artifacts(self, username: str, notebook_id: str) -> list[dict[str, str]]:
        paths = self.notebook_paths(username, notebook_id)
        rows: list[dict[str, str]] = []
        for kind, folder in [("report", paths.reports), ("quiz", paths.quizzes), ("podcast", paths.podcasts)]:
            if not folder.exists():
                continue
            for p in sorted(folder.iterdir()):
                if p.is_file():
                    rows.append({"type": kind, "name": p.name, "path": str(p)})
        return rows

    def _artifact_dir(self, username: str, notebook_id: str, artifact_type: str) -> Path:
        paths = self.notebook_paths(username, notebook_id)
        mapping = {
            "report": paths.reports,
            "quiz": paths.quizzes,
            "podcast": paths.podcasts,
        }
        if artifact_type not in mapping:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
        return mapping[artifact_type]

    def _next_artifact_name(self, artifact_dir: Path, prefix: str, extension: str) -> str:
        ext = extension if extension.startswith(".") else f".{extension}"
        max_idx = 0
        for p in artifact_dir.glob(f"{prefix}_*{ext}"):
            stem = p.stem
            try:
                idx = int(stem.split("_")[-1])
                max_idx = max(max_idx, idx)
            except (ValueError, IndexError):
                continue
        return f"{prefix}_{max_idx + 1}{ext}"
