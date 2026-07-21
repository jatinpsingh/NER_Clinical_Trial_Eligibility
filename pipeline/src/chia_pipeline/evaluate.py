"""Testing pipeline: score any track's predictions against the shared gold test set.

This is the piece Jatin asked for: a single script that takes a model's
predictions and the gold data/processed/{split}_spans.jsonl, lines them up
by sentence id, and reports exact + relaxed entity-level precision/recall/F1
(chia_pipeline.eval_utils), so every track (PubMedBERT, small LMs, frontier
LLMs) is judged the same way.

Different tracks naturally produce different output shapes, so this accepts
two prediction formats rather than forcing everyone to hand-format spans:

  spans  {"id": ..., "entities": [{"type", "start", "end"}, ...]}
         -- what the LLM/prompting tracks naturally produce (they read the
         raw text, so they can report character offsets directly).

  bio    {"id": ..., "tokens": [...], "predicted_tags": [...]} 
         -- what a HuggingFace token-classification model (e.g. PubMedBERT)
         naturally produces: one BIO tag per input token. This is decoded
         into spans via chia_pipeline.align.bio_to_spans, using the *gold*
         sentence's tokenization (data/processed/{split}.jsonl) re-derived
         with the same tokenizer, so offsets line up. If a model's own
tokens don't match ours 1:1 (e.g. subword pieces instead of our
word-level tokens), re-align to word-level tags before calling
this -- see `align_wordpiece_predictions` for the common case.

Usage:
    python -m chia_pipeline.evaluate \
        --gold data/processed/test_spans.jsonl \
        --pred path/to/predictions.jsonl \
        --pred-format spans   # or: bio --tokens data/processed/test.jsonl
"""

import argparse
import json
from pathlib import Path

from .align import bio_to_spans
from .eval_utils import score_corpus_both
from .tokenizer import tokenize


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonicalize_entity(e: dict) -> dict:
    """Keep only the fields eval_utils needs, so extra per-model fields
    (confidence scores, raw text, etc.) don't break anything downstream."""
    return {"type": e["type"], "start": e["start"], "end": e["end"]}


def load_gold(gold_path: Path) -> dict[str, dict]:
    """Return {id: span_record} for every sentence in the gold file."""
    records = _load_jsonl(gold_path)
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError(f"Duplicate ids found in gold file {gold_path}")
    return by_id


def load_predictions_spans(pred_path: Path) -> dict[str, list[dict]]:
    """Load {"id", "entities"} records directly (the LLM/prompting track format)."""
    preds = {}
    for r in _load_jsonl(pred_path):
        preds[r["id"]] = [_canonicalize_entity(e) for e in r["entities"]]
    return preds


def align_wordpiece_predictions(
    gold_tokens: list[str], model_tokens: list[str], model_tags: list[str]
) -> list[str]:
    """Collapse a subword-tokenizer's per-wordpiece tags down to one tag per
    our word-level gold token, by taking the first wordpiece's tag for each
    word (the standard convention for HuggingFace token classification: only
    the first subword of a word carries the "real" prediction).

    Only needed if a model's tokenizer doesn't happen to match `tokenize()`
    word-for-word (e.g. it split on its own subword vocabulary). Requires the
    model's tokens to be a subword split of the same words, in order --
    raises if that assumption doesn't hold, so a mismatch fails loudly
    instead of silently mis-aligning tags.
    """
    tags: list[str] = []
    mi = 0
    for gold_tok in gold_tokens:
        if mi >= len(model_tokens):
            raise ValueError("Ran out of model tokens before matching all gold tokens")
        piece = model_tokens[mi]
        tags.append(model_tags[mi])
        consumed = piece
        mi += 1
        while consumed != gold_tok:
            if mi >= len(model_tokens) or len(consumed) >= len(gold_tok):
                raise ValueError(
                    f"Could not align model tokens to gold token {gold_tok!r} "
                    f"(got {consumed!r}); check the model's tokenizer output."
                )
            consumed += model_tokens[mi].lstrip("#")
            mi += 1
    if mi != len(model_tokens):
        raise ValueError("Model tokens left over after aligning all gold tokens")
    return tags


def load_predictions_bio(
    pred_path: Path, gold_by_id: dict[str, dict], tokens_by_id: dict[str, dict]
) -> dict[str, list[dict]]:
    """Load {"id", "tokens", "predicted_tags"} records and decode to spans.

    Re-tokenizes each gold sentence's text the same way build_dataset.py did,
    and requires the prediction's `tokens` to match exactly -- this is the
    same safety check baselines.py uses, so a silent misalignment (e.g. the
    model dropped punctuation, or tokenized differently) fails loudly instead
    of producing quietly-wrong scores.
    """
    preds = {}
    for r in _load_jsonl(pred_path):
        sid = r["id"]
        if sid not in gold_by_id:
            raise ValueError(f"Prediction id {sid!r} not found in gold file")
        text = gold_by_id[sid]["text"]
        retokenized = tokenize(text)
        expected_tokens = tokens_by_id[sid]["tokens"]
        if [t.text for t in retokenized] != expected_tokens:
            raise ValueError(f"Gold token file and gold text disagree for id {sid!r}")
        if r["tokens"] != expected_tokens:
            raise ValueError(
                f"Prediction tokens for id {sid!r} don't match gold tokens -- "
                "if your model uses its own tokenizer, run "
                "align_wordpiece_predictions() first to collapse to word-level tags."
            )
        spans = bio_to_spans(retokenized, r["predicted_tags"])
        preds[sid] = [_canonicalize_entity(s) for s in spans]
    return preds


def evaluate(
    gold_path: Path,
    pred_path: Path,
    pred_format: str = "spans",
    tokens_path: Path | None = None,
) -> dict:
    """Score predictions against gold. Returns the score_corpus_both() dict,
    plus a `coverage` block reporting any gold sentences a model didn't
    predict for at all (treated as empty predictions, but worth surfacing --
    a model that silently skips sentences shouldn't get a free pass)."""
    gold_by_id = load_gold(gold_path)

    if pred_format == "spans":
        pred_by_id = load_predictions_spans(pred_path)
    elif pred_format == "bio":
        if tokens_path is None:
            raise ValueError("--tokens is required when --pred-format bio")
        tokens_by_id = {r["id"]: r for r in _load_jsonl(tokens_path)}
        pred_by_id = load_predictions_bio(pred_path, gold_by_id, tokens_by_id)
    else:
        raise ValueError(f"Unknown pred_format {pred_format!r}, expected 'spans' or 'bio'")

    missing_ids = sorted(set(gold_by_id) - set(pred_by_id))

    gold_sentences, pred_sentences = [], []
    for sid, gold_record in gold_by_id.items():
        gold_sentences.append(gold_record["entities"])
        pred_sentences.append(pred_by_id.get(sid, []))

    scores = score_corpus_both(gold_sentences, pred_sentences)
    scores["coverage"] = {
        "num_gold_sentences": len(gold_by_id),
        "num_predicted_sentences": len(pred_by_id),
        "num_missing_predictions": len(missing_ids),
        "missing_ids_sample": missing_ids[:10],
    }
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, required=True, help="Gold *_spans.jsonl file")
    parser.add_argument("--pred", type=Path, required=True, help="Predictions jsonl file")
    parser.add_argument("--pred-format", choices=["spans", "bio"], default="spans")
    parser.add_argument("--tokens", type=Path, default=None, help="Gold {split}.jsonl, required for --pred-format bio")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write the report JSON to")
    args = parser.parse_args()

    report = evaluate(args.gold, args.pred, args.pred_format, args.tokens)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
