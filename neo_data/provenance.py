"""Shared provenance logging for Phase 1 corpus acquisition.

Every downloader in neo_data/ calls log_provenance with this exact
signature. Factored into one module instead of duplicating the same
function five times.
"""
import hashlib
import json
import time
from pathlib import Path


def log_provenance(source_url: str, file_hash: str, record_count: int, output_path: str) -> None:
    """Append one entry to neo_data/provenance_log.jsonl."""
    log_path = Path(__file__).parent / "provenance_log.jsonl"
    entry = {
        "source_url": source_url,
        "file_hash": file_hash,
        "record_count": record_count,
        "output_path": str(output_path),
        "timestamp": time.time(),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def hash_file(path: str) -> str:
    """SHA-256 of a file, streamed to avoid loading large downloads into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
