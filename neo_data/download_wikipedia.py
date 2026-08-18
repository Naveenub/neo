"""Phase 1 — Wikipedia corpus acquisition. Direct wikimedia.org dump, no HF."""
import json
import subprocess
import urllib.request
from pathlib import Path

from neo_data.provenance import hash_file, log_provenance

WIKIMEDIA_DUMP_BASE = "https://dumps.wikimedia.org"


def download_wikipedia_dump(lang: str, output_dir: str) -> str:
    """wget the latest enwiki (or other lang) pages-articles dump."""
    url = f"{WIKIMEDIA_DUMP_BASE}/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / f"{lang}wiki-latest-pages-articles.xml.bz2"

    subprocess.run(["wget", "-c", "-O", str(dump_path), url], check=True)
    return str(dump_path)


def extract_wikipedia_text(dump_path: str, output_dir: str) -> str:
    """Run wikiextractor as a subprocess to pull plaintext JSONL from the dump."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "python", "-m", "wikiextractor.WikiExtractor",
            dump_path,
            "-o", str(out_dir),
            "--json",
            "--no-templates",
        ],
        check=True,
    )
    return str(out_dir)


def merge_wikipedia_jsonl(input_dir: str, output_txt: str) -> int:
    """Flatten wikiextractor's nested JSONL output into one filtered .txt file.

    Returns the number of documents written.
    """
    doc_count = 0
    with open(output_txt, "w", encoding="utf-8") as out_f:
        for path in sorted(Path(input_dir).rglob("wiki_*")):
            with open(path, encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    text = doc.get("text", "").strip()
                    if len(text) < 1:
                        continue
                    out_f.write(text + "\n")
                    doc_count += 1
    return doc_count


def main() -> None:
    lang = "en"
    raw_dump_dir = "data/raw/wikipedia_dump"
    extracted_dir = "data/raw/wikipedia_extracted"
    merged_txt = "data/raw/wikipedia.txt"

    dump_path = download_wikipedia_dump(lang, raw_dump_dir)
    extract_wikipedia_text(dump_path, extracted_dir)
    doc_count = merge_wikipedia_jsonl(extracted_dir, merged_txt)

    log_provenance(
        source_url=f"{WIKIMEDIA_DUMP_BASE}/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2",
        file_hash=hash_file(merged_txt),
        record_count=doc_count,
        output_path=merged_txt,
    )
    print(f"Wikipedia: {doc_count} docs -> {merged_txt}")


if __name__ == "__main__":
    main()
