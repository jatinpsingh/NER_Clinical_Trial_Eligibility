"""Exploratory data analysis over the processed CHIA splits."""

import argparse
import json
import statistics as stats_lib
from collections import Counter
from pathlib import Path

from .build_dataset import DEFAULT_OUT_DIR


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def analyze(out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    result = {}
    train_tokens = _load_jsonl(out_dir / "train.jsonl")
    train_spans = _load_jsonl(out_dir / "train_spans.jsonl")
    train_vocab = {tok.lower() for r in train_tokens for tok in r["tokens"]}

    for split in ("train", "val", "test"):
        token_records = _load_jsonl(out_dir / f"{split}.jsonl")
        span_records = _load_jsonl(out_dir / f"{split}_spans.jsonl")

        sent_lengths = [len(r["tokens"]) for r in token_records]
        entity_counts = [len(r["entities"]) for r in span_records]
        by_criteria_type = Counter(r["criteria_type"] for r in span_records)
        entities_per_criteria_type = {}
        for ctype in ("inclusion", "exclusion"):
            subset = [len(r["entities"]) for r in span_records if r["criteria_type"] == ctype]
            entities_per_criteria_type[ctype] = round(sum(subset) / len(subset), 2) if subset else 0.0

        split_vocab = {tok.lower() for r in token_records for tok in r["tokens"]}
        oov_rate = (
            len(split_vocab - train_vocab) / len(split_vocab) if split_vocab and split != "train" else 0.0
        )

        result[split] = {
            "sentences": len(token_records),
            "documents": len({r["nct_id"] for r in token_records}),
            "sentence_len_tokens": {
                "mean": round(stats_lib.mean(sent_lengths), 2),
                "median": stats_lib.median(sent_lengths),
                "p90": stats_lib.quantiles(sent_lengths, n=10)[8],
                "max": max(sent_lengths),
            },
            "entities_per_sentence": {
                "mean": round(stats_lib.mean(entity_counts), 2),
                "zero_entity_sentences_pct": round(
                    100 * sum(1 for c in entity_counts if c == 0) / len(entity_counts), 1
                ),
            },
            "criteria_type_counts": dict(by_criteria_type),
            "mean_entities_per_sentence_by_criteria_type": entities_per_criteria_type,
            "unique_tokens": len(split_vocab),
            "oov_token_rate_vs_train": round(oov_rate, 3),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(analyze(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
