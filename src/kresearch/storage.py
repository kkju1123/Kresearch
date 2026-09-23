"""Local filesystem storage for snapshot text and report markdown.

M1 keeps this simple (plain files on disk, referenced by path in
content_ref) instead of object storage; swapping the backend later only
means changing these two functions.
"""

import hashlib
import uuid
from pathlib import Path

SNAPSHOT_DIR = Path("reports/snapshots")
REPORT_DIR = Path("reports")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_snapshot(snapshot_id: uuid.UUID, text: str) -> str:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{snapshot_id}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def load_snapshot(content_ref: str) -> str:
    return Path(content_ref).read_text(encoding="utf-8")


def save_report(task_id: uuid.UUID, version: int, markdown: str) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{task_id}_v{version}.md"
    path.write_text(markdown, encoding="utf-8")
    return str(path)
