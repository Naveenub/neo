"""Phase 1 — The Stack v2 code subset acquisition. Direct CDN download, no HF."""
import re
import urllib.request
from pathlib import Path

from neo_data.provenance import hash_file, log_provenance

MIN_LINES = 50
MAX_BYTES = 100 * 1024
FUNC_DEF_PATTERNS = {
    "python": re.compile(r"^\s*def\s+\w+\s*\("),
    "c": re.compile(r"^\s*\w[\w\s\*]*\s+\w+\s*\([^;]*\)\s*\{?\s*$"),
    "bash": re.compile(r"^\s*(\w+\s*\(\)\s*\{|function\s+\w+)"),
}


def download_stack_subset(language: str, output_dir: str) -> str:
    """Download a language subset archive from the CDN and unpack into output_dir."""
    cdn_url = f"https://the-stack-cdn.example.org/v2/{language}/subset.tar.gz"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"{language}_subset.tar.gz"
    urllib.request.urlretrieve(cdn_url, archive_path)

    import tarfile
    with tarfile.open(archive_path) as tar:
        tar.extractall(out_dir)
    return str(out_dir)


def filter_code_file(content: str) -> bool:
    """True = keep. Reject too-short, too-large, or files with no function definitions."""
    if len(content.encode("utf-8")) > MAX_BYTES:
        return False
    lines = content.splitlines()
    if len(lines) < MIN_LINES:
        return False
    has_func = any(
        pattern.search(line)
        for pattern in FUNC_DEF_PATTERNS.values()
        for line in lines
    )
    return has_func


def normalize_code(content: str) -> str:
    """Strip trailing whitespace per line, normalize line endings to \\n, remove BOM."""
    content = content.lstrip("\ufeff")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in content.split("\n")]
    return "\n".join(lines)


def main(language: str = "python") -> None:
    raw_dir = f"data/raw/stack_{language}_raw"
    out_path = f"data/raw/stack_{language}.txt"

    download_stack_subset(language, raw_dir)

    kept = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for path in Path(raw_dir).rglob("*"):
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except (UnicodeDecodeError, OSError):
                continue
            if not filter_code_file(content):
                continue
            normalized = normalize_code(content)
            out_f.write(normalized + "\n" + ("-" * 40) + "\n")
            kept += 1

    log_provenance(
        source_url=f"https://the-stack-cdn.example.org/v2/{language}/subset.tar.gz",
        file_hash=hash_file(out_path),
        record_count=kept,
        output_path=out_path,
    )
    print(f"Stack ({language}): {kept} files kept -> {out_path}")


if __name__ == "__main__":
    main()
