"""Lightweight entity-level F1 for per-epoch validation monitoring during training.

This is deliberately self-contained (no chia_pipeline dependency, no character
offsets) -- it decodes BIO tag sequences into (type, start_word, end_word) spans
and does exact-match scoring, which is equivalent to eval_utils.score_corpus's
char-offset exact match for the same tokenization (word index and char offset
are just two coordinate systems for the same boundaries). It exists only to let
the Trainer pick a "best" checkpoint each epoch without needing to fetch/retokenize
the *_spans.jsonl gold text on every evaluation call.

The authoritative, reported entity-level score (strict, and relaxed once a
teammate's score_corpus_relaxed lands) is computed separately in evaluate.py
using chia_pipeline.eval_utils, against the *_spans.jsonl char-offset gold data.
"""

from collections import Counter


def bio_tags_to_word_spans(tags: list[str]) -> list[tuple[str, int, int]]:
    """Decode a list of BIO tags into (type, start_word_idx, end_word_idx) spans."""
    spans = []
    current_type = None
    current_start = None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_type is not None:
                spans.append((current_type, current_start, i))
            current_type = tag[2:]
            current_start = i
        elif tag.startswith("I-") and current_type == tag[2:]:
            continue
        else:
            if current_type is not None:
                spans.append((current_type, current_start, i))
            current_type = None
            current_start = None
    if current_type is not None:
        spans.append((current_type, current_start, len(tags)))
    return spans


def _prf(tp: int, gold_count: int, pred_count: int) -> dict:
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": gold_count}


def word_level_entity_f1(gold_tags_list: list[list[str]], pred_tags_list: list[list[str]]) -> dict:
    """Strict (exact word-span + type) micro entity-level P/R/F1 across a corpus."""
    tp_total = gold_total = pred_total = 0
    for gold_tags, pred_tags in zip(gold_tags_list, pred_tags_list):
        gold_spans = Counter(bio_tags_to_word_spans(gold_tags))
        pred_spans = Counter(bio_tags_to_word_spans(pred_tags))
        tp = gold_spans & pred_spans
        tp_total += sum(tp.values())
        gold_total += sum(gold_spans.values())
        pred_total += sum(pred_spans.values())
    return _prf(tp_total, gold_total, pred_total)
