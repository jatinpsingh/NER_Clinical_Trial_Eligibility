"""Print entity-type and split statistics for the processed CHIA dataset."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .build_dataset import DEFAULT_OUT_DIR


def summarize(out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    summary = {}
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}_spans.jsonl"
        if not path.exists():
            continue
        sentences = 0
        nct_ids = set()
        type_counts = Counter()
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            sentences += 1
            nct_ids.add(record["nct_id"])
            type_counts.update(e["type"] for e in record["entities"])
        summary[split] = {
            "sentences": sentences,
            "documents": len(nct_ids),
            "entities_total": sum(type_counts.values()),
            "entities_by_type": dict(type_counts.most_common()),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(summarize(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
