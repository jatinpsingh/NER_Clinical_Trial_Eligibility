"""Entity-level precision/recall/F1, shared by the PubMedBERT and GPT-4 pipelines.

Using one scorer for both models is what makes the comparison in the paper
fair: both are judged against the same flattened gold spans (data/processed/
*_spans.jsonl).

Two matching conventions are supported, matching what Li et al. (2022)
report for PubMedBERT on CHIA (0.622 strict / 0.744 relaxed), so our numbers
are directly comparable to theirs:

  - exact (score_corpus / score_sentence): a predicted entity only counts if
    its (type, start, end) is character-for-character identical to a gold
    entity. Boundary mistakes (e.g. missing a leading "type 2") count as a
    full miss.
  - relaxed (score_corpus_relaxed / score_sentence_relaxed): a predicted
    entity counts if it has the same type as a gold entity AND their spans
    overlap at all. Forgiving of boundary mistakes, punishing only wrong
type or no overlap whatsoever.
"""

from collections import Counter, defaultdict


def _as_key(entity: dict) -> tuple:
    return (entity["type"], entity["start"], entity["end"])


def _overlaps(a: dict, b: dict) -> bool:
    """Whether two entities' character spans share at least one character."""
    return a["start"] < b["end"] and b["start"] < a["end"]


def prf(tp: int, gold_count: int, pred_count: int) -> dict:
    """Precision/recall/F1 from raw counts. 0.0 (not an error) when a
    denominator is 0 -- e.g. a type with no gold mentions in this split."""
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": gold_count}


def score_sentence(gold: list[dict], pred: list[dict]) -> tuple[Counter, Counter, Counter]:
    """Return (tp_by_type, gold_by_type, pred_by_type) counters for one
    sentence, under exact matching: a predicted entity only counts if its
    (type, start, end) is identical to a gold entity."""
    gold_keys = Counter(_as_key(e) for e in gold)
    pred_keys = Counter(_as_key(e) for e in pred)
    tp_keys = gold_keys & pred_keys

    tp_by_type: Counter = Counter()
    for (etype, _start, _end), count in tp_keys.items():
        tp_by_type[etype] += count
    gold_by_type = Counter(e["type"] for e in gold)
    pred_by_type = Counter(e["type"] for e in pred)
    return tp_by_type, gold_by_type, pred_by_type


def score_sentence_relaxed(gold: list[dict], pred: list[dict]) -> tuple[Counter, Counter, Counter]:
    """Return (tp_by_type, gold_by_type, pred_by_type) for one sentence,
    under relaxed (same-type, any-overlap) matching.

    Matching is greedy and one-to-one *within each type*: each gold entity
    can be satisfied by at most one predicted entity and vice versa, so one
    long predicted span can't count as a match against several short gold
    spans it happens to overlap (or vice versa). Ties (equal overlap length)
    are broken by gold order, which is deterministic given the same input.
    """
    gold_by_type: Counter = Counter(e["type"] for e in gold)
    pred_by_type: Counter = Counter(e["type"] for e in pred)
    tp_by_type: Counter = Counter()

    gold_by_key_type: dict[str, list[dict]] = defaultdict(list)
    for e in gold:
        gold_by_key_type[e["type"]].append(e)
    pred_by_key_type: dict[str, list[dict]] = defaultdict(list)
    for e in pred:
        pred_by_key_type[e["type"]].append(e)

    for etype, gold_ents in gold_by_key_type.items():
        pred_ents = pred_by_key_type.get(etype, [])
        if not pred_ents:
            continue
        candidates = []
        for gi, g in enumerate(gold_ents):
            for pi, p in enumerate(pred_ents):
                if _overlaps(g, p):
                    overlap_len = min(g["end"], p["end"]) - max(g["start"], p["start"])
                    candidates.append((overlap_len, gi, pi))
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_gold: set[int] = set()
        matched_pred: set[int] = set()
        for _overlap_len, gi, pi in candidates:
            if gi in matched_gold or pi in matched_pred:
                continue
            matched_gold.add(gi)
            matched_pred.add(pi)
        tp_by_type[etype] += len(matched_gold)

    return tp_by_type, gold_by_type, pred_by_type


def _score_corpus(
    gold_sentences: list[list[dict]],
    pred_sentences: list[list[dict]],
    sentence_scorer,
) -> dict:
    """Shared aggregation logic for both matching conventions."""
    if len(gold_sentences) != len(pred_sentences):
        raise ValueError("gold and pred must have the same number of sentences")

    tp_by_type: Counter = Counter()
    gold_by_type: Counter = Counter()
    pred_by_type: Counter = Counter()

    for gold, pred in zip(gold_sentences, pred_sentences):
        tp, gold_keys, pred_keys = sentence_scorer(gold, pred)
        tp_by_type.update(tp)
        gold_by_type.update(gold_keys)
        pred_by_type.update(pred_keys)

    per_type = {
        etype: prf(tp_by_type[etype], gold_by_type[etype], pred_by_type[etype])
        for etype in sorted(set(gold_by_type) | set(pred_by_type))
    }
    overall = prf(sum(tp_by_type.values()), sum(gold_by_type.values()), sum(pred_by_type.values()))
    return {"overall": overall, "per_type": per_type}


def score_corpus(gold_sentences: list[list[dict]], pred_sentences: list[list[dict]]) -> dict:
    """Micro-averaged and per-type precision/recall/F1 under exact matching.

    gold_sentences / pred_sentences: one list of entity dicts
    ({"type", "start", "end"}) per sentence, in matching order.
    """
    return _score_corpus(gold_sentences, pred_sentences, score_sentence)


def score_corpus_relaxed(gold_sentences: list[list[dict]], pred_sentences: list[list[dict]]) -> dict:
    """Micro-averaged and per-type precision/recall/F1 under relaxed
    (same-type, any-overlap) matching. Same input shape as score_corpus."""
    return _score_corpus(gold_sentences, pred_sentences, score_sentence_relaxed)


def score_corpus_both(gold_sentences: list[list[dict]], pred_sentences: list[list[dict]]) -> dict:
    """Convenience wrapper returning {"exact": ..., "relaxed": ...} so callers
    (e.g. the evaluate CLI) get both metrics from one call."""
    return {
        "exact": score_corpus(gold_sentences, pred_sentences),
        "relaxed": score_corpus_relaxed(gold_sentences, pred_sentences),
    }
