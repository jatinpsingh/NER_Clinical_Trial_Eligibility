"""Turn raw CHIA BRAT files into split, model-ready NER datasets.

Each line of a CHIA .txt file is one eligibility criterion (already
sentence-segmented by the annotators), so we treat lines as sentences
directly rather than running a separate sentence splitter.

Outputs two parallel formats per split, sharing the same flattened gold
entities so PubMedBERT and GPT-4 are scored against identical gold spans:
  - {split}.jsonl        tokens + BIO tags, for HuggingFace token classification
  - {split}_spans.jsonl  raw text + char-offset entity spans, for LLM prompting/eval
"""

import argparse
import json
import random
from pathlib import Path

from .align import bio_to_spans, fragments_to_bio, resolve_overlaps
from .brat import Fragment, parse_ann
from .constants import LABELS, SPLIT_SEED, TRAIN_FRAC, VAL_FRAC
from .tokenizer import tokenize

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PIPELINE_ROOT.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chia_without_scope"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"


def discover_documents(raw_dir: Path) -> list[tuple[str, str, Path, Path]]:
    """Return (nct_id, criteria_type, txt_path, ann_path) for every doc pair."""
    docs = []
    for txt_path in sorted(raw_dir.glob("*.txt")):
        stem = txt_path.stem
        if stem.endswith("_inc"):
            nct_id, criteria_type = stem[: -len("_inc")], "inclusion"
        elif stem.endswith("_exc"):
            nct_id, criteria_type = stem[: -len("_exc")], "exclusion"
        else:
            continue
        ann_path = txt_path.with_suffix(".ann")
        if ann_path.exists():
            docs.append((nct_id, criteria_type, txt_path, ann_path))
    return docs


def iter_sentences(doc_text: str):
    """Yield (sentence_text, line_start_offset) for each non-blank line."""
    offset = 0
    for line in doc_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            yield stripped, offset + (len(line) - len(line.lstrip()))
        offset += len(line)


def build_sentence_record(sentence_text: str, line_start: int, fragments: list[Fragment]) -> dict:
    local_fragments = [
        Fragment(f.entity_id, f.type, f.start - line_start, f.end - line_start)
        for f in fragments
        if line_start <= f.start and f.end <= line_start + len(sentence_text)
    ]
    tokens = tokenize(sentence_text)
    kept_fragments = resolve_overlaps(local_fragments)
    tags = fragments_to_bio(tokens, kept_fragments)
    spans = bio_to_spans(tokens, tags)
    for span in spans:
        span["text"] = sentence_text[span["start"] : span["end"]]
    return {
        "text": sentence_text,
        "tokens": [t.text for t in tokens],
        "ner_tags": tags,
        "entities": spans,
        "num_dropped_overlaps": len(local_fragments) - len(kept_fragments),
    }


def process_document(nct_id: str, criteria_type: str, txt_path: Path, ann_path: Path) -> list[dict]:
    doc_text = txt_path.read_text(encoding="utf-8")
    fragments = parse_ann(ann_path.read_text(encoding="utf-8"))
    records = []
    for i, (sentence_text, line_start) in enumerate(iter_sentences(doc_text)):
        record = build_sentence_record(sentence_text, line_start, fragments)
        record.update(
            id=f"{nct_id}_{criteria_type[:3]}_{i}",
            nct_id=nct_id,
            criteria_type=criteria_type,
        )
        records.append(record)
    return records


def split_doc_ids(nct_ids: list[str]) -> dict[str, str]:
    shuffled = sorted(set(nct_ids))
    random.Random(SPLIT_SEED).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val = n_train + int(n * VAL_FRAC)
    assignment = {}
    for doc_id in shuffled[:n_train]:
        assignment[doc_id] = "train"
    for doc_id in shuffled[n_train:n_val]:
        assignment[doc_id] = "val"
    for doc_id in shuffled[n_val:]:
        assignment[doc_id] = "test"
    return assignment


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def build(raw_dir: Path = DEFAULT_RAW_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    docs = discover_documents(raw_dir)
    if not docs:
        raise FileNotFoundError(f"No .txt/.ann pairs found under {raw_dir}")

    all_records = []
    for nct_id, criteria_type, txt_path, ann_path in docs:
        all_records.extend(process_document(nct_id, criteria_type, txt_path, ann_path))

    split_by_doc = split_doc_ids([r["nct_id"] for r in all_records])
    for r in all_records:
        r["split"] = split_by_doc[r["nct_id"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        split_records = [r for r in all_records if r["split"] == split]
        counts[split] = len(split_records)

        token_records = [
            {
                "id": r["id"],
                "nct_id": r["nct_id"],
                "criteria_type": r["criteria_type"],
                "tokens": r["tokens"],
                "ner_tags": r["ner_tags"],
            }
            for r in split_records
        ]
        write_jsonl(out_dir / f"{split}.jsonl", token_records)

        span_records = [
            {
                "id": r["id"],
                "nct_id": r["nct_id"],
                "criteria_type": r["criteria_type"],
                "text": r["text"],
                "entities": r["entities"],
            }
            for r in split_records
        ]
        write_jsonl(out_dir / f"{split}_spans.jsonl", span_records)

    (out_dir / "label_list.json").write_text(
        json.dumps(
            {"labels": LABELS, "label2id": {l: i for i, l in enumerate(LABELS)}},
            indent=2,
        )
    )

    return {
        "num_documents": len(docs),
        "num_sentences": len(all_records),
        "split_counts": counts,
        "num_overlaps_dropped": sum(r["num_dropped_overlaps"] for r in all_records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = build(args.raw_dir, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
