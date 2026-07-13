"""Entity-level precision/recall/F1, shared by the PubMedBERT and GPT-4 pipelines.

Using one scorer for both models is what makes the comparison in the paper
fair: both are judged against the same flattened gold spans (data/processed/
*_spans.jsonl), with exact (type, start, end) match — the standard
entity-level scoring convention used in prior CHIA NER work.
"""

from collections import Counter


def _as_key(entity: dict) -> tuple:
    return (entity["type"], entity["start"], entity["end"])


def score_sentence(gold: list[dict], pred: list[dict]) -> tuple[Counter, Counter]:
    """Return (tp, fp_fn_support) counters keyed by entity type for one sentence."""
    gold_keys = Counter(_as_key(e) for e in gold)
    pred_keys = Counter(_as_key(e) for e in pred)
    tp = gold_keys & pred_keys
    return tp, gold_keys, pred_keys


def score_corpus(gold_sentences: list[list[dict]], pred_sentences: list[list[dict]]) -> dict:
    """Compute micro-averaged and per-type precision/recall/F1.

    gold_sentences / pred_sentences: one list of entity dicts
    ({"type", "start", "end"}) per sentence, in matching order.
    """
    if len(gold_sentences) != len(pred_sentences):
        raise ValueError("gold and pred must have the same number of sentences")

    tp_by_type: Counter = Counter()
    gold_by_type: Counter = Counter()
    pred_by_type: Counter = Counter()

    for gold, pred in zip(gold_sentences, pred_sentences):
        tp, gold_keys, pred_keys = score_sentence(gold, pred)
        for (etype, _start, _end), count in tp.items():
            tp_by_type[etype] += count
        for (etype, _start, _end), count in gold_keys.items():
            gold_by_type[etype] += count
        for (etype, _start, _end), count in pred_keys.items():
            pred_by_type[etype] += count

    def prf(tp: int, gold_count: int, pred_count: int) -> dict:
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "support": gold_count}

    per_type = {
        etype: prf(tp_by_type[etype], gold_by_type[etype], pred_by_type[etype])
        for etype in sorted(set(gold_by_type) | set(pred_by_type))
    }
    overall = prf(sum(tp_by_type.values()), sum(gold_by_type.values()), sum(pred_by_type.values()))

    return {"overall": overall, "per_type": per_type}
