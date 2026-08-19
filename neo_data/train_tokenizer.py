"""Phase 2 — train Neo's own BPE tokenizer from the Phase 1 corpus. No pre-built vocab."""
import json
import random
from pathlib import Path

import sentencepiece as spm

VOCAB_SIZE = 32000
SPECIAL_TOKENS = [
    "<|neo|>", "<|end|>", "<|system|>", "<|user|>",
    "<|assistant|>", "<|tool|>", "<|pad|>",
]
BOS_PIECE = "<|neo|>"
EOS_PIECE = "<|end|>"


def merge_corpus_for_tokenizer(source_dirs: list[str], output_path: str, max_sentences: int) -> str:
    """Sample sentences from all sources proportionally; write to one flat file for spm training."""
    files: list[Path] = []
    for d in source_dirs:
        files.extend(Path(d).rglob("*.txt"))
        files.extend(Path(d).rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no corpus files found under {source_dirs}")

    per_file_quota = max(1, max_sentences // len(files))
    sentences: list[str] = []
    for path in files:
        lines = _read_lines(path)
        random.shuffle(lines)
        sentences.extend(lines[:per_file_quota])

    random.shuffle(sentences)
    sentences = sentences[:max_sentences]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))
    return output_path


def _read_lines(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = doc.get("body_text") or doc.get("text") or ""
                out.extend(s.strip() for s in text.splitlines() if s.strip())
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def train_bpe_tokenizer(corpus_path: str, output_prefix: str, vocab_size: int, special_tokens: list[str]) -> str:
    """Train a SentencePiece BPE model with Neo's exact hyperparameters."""
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        bos_piece=BOS_PIECE,
        eos_piece=EOS_PIECE,
        user_defined_symbols=special_tokens,
    )
    return f"{output_prefix}.model"


def validate_tokenizer(model_path: str, test_strings: list[str]) -> dict:
    """Encode/decode round-trip on test strings; report fertility (tokens/word)."""
    sp = spm.SentencePieceProcessor(model_file=model_path)
    exact_matches = 0
    total_tokens = 0
    total_words = 0
    for text in test_strings:
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        if decoded.strip() == text.strip():
            exact_matches += 1
        total_tokens += len(ids)
        total_words += max(1, len(text.split()))

    fertility = total_tokens / total_words
    result = {
        "roundtrip_accuracy": exact_matches / len(test_strings),
        "fertility": fertility,
        "vocab_size": sp.vocab_size(),
    }
    assert fertility < 2.0, f"fertility {fertility:.3f} >= 2.0"
    return result


def export_tokenizer_config(model_path: str, output_json: str) -> str:
    """Write vocab_size, special token IDs, and model path to JSON."""
    sp = spm.SentencePieceProcessor(model_file=model_path)
    config = {
        "model_path": str(model_path),
        "vocab_size": sp.vocab_size(),
        "special_token_ids": {tok: sp.piece_to_id(tok) for tok in SPECIAL_TOKENS},
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return output_json


def main() -> None:
    corpus_path = merge_corpus_for_tokenizer(
        source_dirs=["data/raw"],
        output_path="data/raw/tokenizer_corpus.txt",
        max_sentences=2_000_000,
    )
    model_path = train_bpe_tokenizer(
        corpus_path=corpus_path,
        output_prefix="neo_model/tokenizer/neo",
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )
    test_strings = [line for line in _read_lines(Path(corpus_path))[:50]]
    report = validate_tokenizer(model_path, test_strings)
    print(f"round-trip={report['roundtrip_accuracy']:.3f} fertility={report['fertility']:.3f}")
    export_tokenizer_config(model_path, "neo_model/tokenizer/neo_config.json")


if __name__ == "__main__":
    main()
