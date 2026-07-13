"""Simple, non-neural baselines to sanity-check the pipeline before PubMedBERT/GPT-4.

Two baselines, both trained/evaluated purely on the processed jsonl files:
  - all-O: predicts no entities anywhere (lower bound).
  - lookup: memorizes each token's most frequent BIO tag in train; unseen
    tokens default to O. A standard "how much is pure memorization worth"
    baseline for NER (c.f. CoNLL-2003 frequency baselines) — any trained
    model should clear this by a wide margin, especially on entity types
    that depend on surrounding context (Negation, Value, Qualifier).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .align import bio_to_spans
from .build_dataset import DEFAULT_OUT_DIR
from .eval_utils import score_corpus
from .tokenizer import tokenize


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def train_lookup_baseline(train_token_records: list[dict]) -> dict[str, str]:
    tag_counts: dict[str, Counter] = defaultdict(Counter)
    for record in train_token_records:
        for token, tag in zip(record["tokens"], record["ner_tags"]):
            tag_counts[token.lower()][tag] += 1
    return {token: counts.most_common(1)[0][0] for token, counts in tag_counts.items()}


def _predict_spans(text: str, tokens: list[str], majority_tag: dict[str, str]) -> list[dict]:
    retokenized = tokenize(text)
    assert [t.text for t in retokenized] == tokens, "retokenization mismatch"
    predicted_tags = [majority_tag.get(tok.lower(), "O") for tok in tokens]
    return bio_to_spans(retokenized, predicted_tags)


def evaluate_split(
    split: str, majority_tag: dict[str, str] | None, out_dir: Path = DEFAULT_OUT_DIR
) -> dict:
    token_records = {r["id"]: r for r in _load_jsonl(out_dir / f"{split}.jsonl")}
    span_records = _load_jsonl(out_dir / f"{split}_spans.jsonl")

    gold_sentences, pred_sentences = [], []
    for span_record in span_records:
        token_record = token_records[span_record["id"]]
        gold_sentences.append(span_record["entities"])
        if majority_tag is None:
            pred_sentences.append([])  # all-O baseline
        else:
            pred_sentences.append(
                _predict_spans(span_record["text"], token_record["tokens"], majority_tag)
            )
    return score_corpus(gold_sentences, pred_sentences)


def run(out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    train_token_records = _load_jsonl(out_dir / "train.jsonl")
    majority_tag = train_lookup_baseline(train_token_records)

    results = {}
    for split in ("val", "test"):
        results.setdefault(split, {})["all_O"] = evaluate_split(split, None, out_dir)
        results.setdefault(split, {})["lookup"] = evaluate_split(split, majority_tag, out_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
