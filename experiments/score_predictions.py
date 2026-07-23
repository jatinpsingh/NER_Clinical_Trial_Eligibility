"""Score the prompted-arm prediction files against gold, reusing the shared
`chia_pipeline.eval_utils` scorer so our numbers are directly comparable to the
PubMedBERT / frontier-LLM tracks and to Li et al. 2022 (0.622 strict / 0.744 relaxed).

Why not call `chia_pipeline.evaluate` directly: that module imports `.align` and
`.tokenizer` at module top for its BIO path, and those files are currently absent
from the working tree, so importing it fails. We only need the spans path, which
delegates to `eval_utils.score_corpus_both` — so we import that engine directly
and do the id-alignment + coverage reporting here (mirroring evaluate.evaluate()).

Our prediction files store gold under "entities" and the model output under "pred",
so we read "pred" (pointing evaluate.py's --pred at these files would score the
gold copy against itself and report a perfect 1.0).

Usage:
    python experiments/score_predictions.py                 # score all outputs/predictions_*.jsonl
    python experiments/score_predictions.py outputs/predictions_gemma4-e2b_n1_mfull_seed42.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/sjatinpal2/NER_Clinical_Trial_Eligibility")
sys.path.insert(0, str(ROOT / "pipeline" / "src"))
from chia_pipeline.eval_utils import score_corpus_both  # noqa: E402

GOLD_PATH = ROOT / "data" / "processed_baseline" / "val_spans.jsonl"
OUT_DIR = ROOT / "experiments" / "outputs"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def score_file(pred_path: Path, gold_by_id: dict[str, dict]) -> dict:
    pred_by_id = {r["id"]: r.get("pred", []) for r in load_jsonl(pred_path)}
    missing = sorted(set(gold_by_id) - set(pred_by_id))

    gold_sentences, pred_sentences = [], []
    for sid, gold_record in gold_by_id.items():
        gold_sentences.append(gold_record["entities"])
        pred_sentences.append(pred_by_id.get(sid, []))

    scores = score_corpus_both(gold_sentences, pred_sentences)
    scores["coverage"] = {
        "num_gold": len(gold_by_id),
        "num_predicted": len(pred_by_id),
        "num_missing": len(missing),
        "missing_sample": missing[:10],
    }
    return scores


def n_from_name(path: Path) -> int:
    m = re.search(r"_n(\d+)_", path.name)
    return int(m.group(1)) if m else -1


def main() -> None:
    gold_by_id = {r["id"]: r for r in load_jsonl(GOLD_PATH)}

    args = sys.argv[1:]
    files = [Path(a) for a in args] if args else sorted(
        OUT_DIR.glob("predictions_*_mfull_seed*.jsonl"), key=n_from_name
    )
    if not files:
        sys.exit("no prediction files found")

    rows = []
    for f in files:
        s = score_file(f, gold_by_id)
        rows.append((f, s))
        n = n_from_name(f)
        ex, rel = s["exact"]["overall"], s["relaxed"]["overall"]
        cov = s["coverage"]
        print(f"\n===== {f.name}  (n={n}) =====")
        print(f"coverage: {cov['num_predicted']}/{cov['num_gold']} predicted, {cov['num_missing']} missing")
        print(f"  STRICT   P={ex['precision']:.3f}  R={ex['recall']:.3f}  F1={ex['f1']:.3f}")
        print(f"  RELAXED  P={rel['precision']:.3f}  R={rel['recall']:.3f}  F1={rel['f1']:.3f}")

    # compact cross-n comparison
    print("\n\n===== micro F1 across runs =====")
    print(f"{'n':>4}  {'strict_P':>8} {'strict_R':>8} {'strict_F1':>9}  {'relax_P':>8} {'relax_R':>8} {'relax_F1':>9}")
    for f, s in sorted(rows, key=lambda r: n_from_name(r[0])):
        ex, rel = s["exact"]["overall"], s["relaxed"]["overall"]
        print(f"{n_from_name(f):>4}  {ex['precision']:>8.3f} {ex['recall']:>8.3f} {ex['f1']:>9.3f}  "
              f"{rel['precision']:>8.3f} {rel['recall']:>8.3f} {rel['f1']:>9.3f}")

    # per-type strict F1 for the best (highest strict F1) run
    best_f, best_s = max(rows, key=lambda r: r[1]["exact"]["overall"]["f1"])
    print(f"\n===== per-type STRICT F1 — best run: {best_f.name} =====")
    print(f"{'type':>16}  {'P':>6} {'R':>6} {'F1':>6} {'support':>8}")
    for etype, m in sorted(best_s["exact"]["per_type"].items(), key=lambda kv: -kv[1]["support"]):
        print(f"{etype:>16}  {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>8}")


if __name__ == "__main__":
    main()
