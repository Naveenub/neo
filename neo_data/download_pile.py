"""Phase 1 — The Pile corpus acquisition. Direct .jsonl.zst shard download, no HF."""
import json
import urllib.request
from pathlib import Path

import zstandard as zstd

from neo_data.provenance import hash_file, log_provenance

MIN_DOC_CHARS = 100
MAX_PERPLEXITY = 1000.0  # placeholder threshold; a real LM scorer plugs in here


def download_pile_shard(shard_url: str, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(shard_url, output_path)
    return output_path


def decompress_zst(zst_path: str, output_txt: str) -> str:
    """Stream-decompress a .zst shard to plain text, one JSON line at a time."""
    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as ifh, open(output_txt, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)
    return output_txt


def filter_pile_doc(doc: dict) -> bool:
    """True = keep. Reject short docs, non-English, or over the perplexity threshold."""
    text = doc.get("text", "")
    if len(text) < MIN_DOC_CHARS:
        return False
    meta = doc.get("meta", {})
    lang = meta.get("lang", "en")
    if lang != "en":
        return False
    perplexity = meta.get("perplexity")
    if perplexity is not None and perplexity > MAX_PERPLEXITY:
        return False
    return True


def process_pile_shard(zst_path: str, output_txt: str) -> int:
    """Decompress + filter + write in one streaming pass. Returns kept doc count."""
    dctx = zstd.ZstdDecompressor()
    kept = 0
    with open(zst_path, "rb") as ifh, open(output_txt, "w", encoding="utf-8") as ofh:
        with dctx.stream_reader(ifh) as reader:
            buffer = b""
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    if filter_pile_doc(doc):
                        ofh.write(doc["text"].strip() + "\n")
                        kept += 1
    return kept


def main(shard_urls: list[str]) -> None:
    for i, url in enumerate(shard_urls):
        zst_path = f"data/raw/pile_shard_{i}.jsonl.zst"
        output_txt = f"data/raw/pile_shard_{i}.txt"
        download_pile_shard(url, zst_path)
        kept = process_pile_shard(zst_path, output_txt)
        log_provenance(
            source_url=url,
            file_hash=hash_file(output_txt),
            record_count=kept,
            output_path=output_txt,
        )
        print(f"Pile shard {i}: {kept} docs kept -> {output_txt}")


if __name__ == "__main__":
    main(shard_urls=[])  # populate with actual shard URLs before running
