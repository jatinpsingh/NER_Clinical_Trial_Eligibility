"""Score a fine-tuned checkpoint against the shared gold spans.

Uses chia_pipeline (installed from the team's shared repo) for tokenization,
BIO->span decoding, and the shared entity-level scorer (score_corpus /
score_corpus_relaxed) -- so these numbers are directly comparable to any
other track's results, all scored the same way. Also writes each split's
predictions to {split}_predictions.jsonl in the {"id", "entities"} format
chia_pipeline.evaluate (the shared CLI) expects, so results here can be
independently re-verified with:

    python -m chia_pipeline.evaluate --gold data_cache/test_spans.jsonl \
        --pred outputs/full_run/test_predictions.jsonl --pred-format spans

Usage:
    python evaluate.py --checkpoint outputs/full_run/final --split both
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from chia_pipeline import eval_utils
from chia_pipeline.align import bio_to_spans
from chia_pipeline.tokenizer import tokenize

import config
from data import decode_predictions_to_word_tags, ensure_data, load_jsonl


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict_batch(model, tokenizer, batch_tokens: list[list[str]], device, max_length: int) -> list[list[str]]:
    encoded = tokenizer(
        batch_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encoded).logits
    predicted_ids = logits.argmax(dim=-1).cpu().tolist()

    id2label = model.config.id2label
    results = []
    for i in range(len(batch_tokens)):
        word_ids = encoded.word_ids(batch_index=i)
        results.append(decode_predictions_to_word_tags(word_ids, predicted_ids[i], id2label))
    return results


def evaluate_split(split: str, model, tokenizer, device, paths: dict[str, Path], batch_size: int = 16) -> dict:
    token_records = {r["id"]: r for r in load_jsonl(paths[split])}
    span_records = load_jsonl(paths[f"{split}_spans"])

    ids, gold_sentences, pred_sentences = [], [], []
    for start in range(0, len(span_records), batch_size):
        batch = span_records[start : start + batch_size]
        batch_tokens = [token_records[r["id"]]["tokens"] for r in batch]
        batch_pred_tags = predict_batch(model, tokenizer, batch_tokens, device, config.MAX_SEQ_LENGTH)

        for span_record, tokens, pred_tags in zip(batch, batch_tokens, batch_pred_tags):
            retokenized = tokenize(span_record["text"])
            assert [t.text for t in retokenized] == tokens, (
                f"retokenization mismatch for {span_record['id']}"
            )
            ids.append(span_record["id"])
            gold_sentences.append(span_record["entities"])
            pred_sentences.append(bio_to_spans(retokenized, pred_tags))

    predictions_path = config.OUTPUT_DIR / "full_run" / f"{split}_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for sid, pred in zip(ids, pred_sentences):
            f.write(json.dumps({"id": sid, "entities": pred}) + "\n")

    return {
        "strict": eval_utils.score_corpus(gold_sentences, pred_sentences),
        "relaxed": eval_utils.score_corpus_relaxed(gold_sentences, pred_sentences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=str(config.OUTPUT_DIR / "full_run" / "final"))
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()

    device = get_device()
    print(f"Loading checkpoint from {args.checkpoint} on {device}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    model.to(device).eval()

    paths = ensure_data()
    splits = ["val", "test"] if args.split == "both" else [args.split]

    results = {}
    for split in splits:
        print(f"Scoring {split}...", file=sys.stderr)
        results[split] = evaluate_split(split, model, tokenizer, device, paths)

    config.RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {config.RESULTS_PATH}", file=sys.stderr)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
