"""Gemma prompted-arm runner on the CHIA TEST split.

Same job as experiment_gemma.py (which runs val), with two differences:
  1. it reads data/processed_baseline/test_spans.jsonl and writes to test_output/,
  2. it scores itself when the run finishes — strict + relaxed micro P/R/F1 and
     per-type strict P/R/F1 — so a test run is one command.

Everything that touches the model (SYSTEM prompt, few-shot superset, JSON schema,
locate(), resume, output row layout) is imported from experiment_gemma rather than
copied, so the test-time prompt is byte-identical to the val-time prompt. Do not
re-declare any of it here: the val→test comparison is only meaningful if the two
runners differ solely in which split they read.

Paths resolve relative to this file, so the script works unchanged on any checkout
regardless of where the repo lives or what the containing directory is called.

Usage:
    python test_gemma.py --model gemma4:e4b -n 500
    python test_gemma.py --model gemma4:e4b -n 500 -m 100      # 100-row smoke test
    python test_gemma.py --model gemma4:e4b -n 500 --score-only  # re-score, no inference
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "pipeline" / "src"))

from experiment_gemma import check_ollama, predict  # noqa: E402
from chia_pipeline.eval_utils import score_corpus_both  # noqa: E402

SUPERSET = HERE / "example_superset_seed42.json"
TEST_SPANS = ROOT / "data" / "processed_baseline" / "test_spans.jsonl"
OUT_DIR = HERE / "test_output"

# the 15 without_scope types; kept separate from the runner's list only so a junk
# type from the model lands in one "_invalid" per-type row instead of its own.
ENTITY_TYPES = {
    "Person", "Condition", "Drug", "Observation", "Measurement", "Procedure",
    "Device", "Visit", "Negation", "Qualifier", "Temporal", "Value",
    "Multiplier", "Reference_point", "Mood",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_examples(n: int) -> list[dict]:
    data = json.loads(SUPERSET.read_text())
    ordered = [data[str(i)] for i in range(len(data))]
    return ordered[:n]


def load_test(m: int | None, seed: int, shuffle: bool) -> list[dict]:
    rows = load_jsonl(TEST_SPANS)
    if shuffle:
        random.Random(seed).shuffle(rows)
    return rows if m is None else rows[:m]


def dropped_as_fp(dropped: list[dict]) -> list[dict]:
    """Turn each unplaceable model output into a guaranteed false positive.

    Identical convention to score_predictions.py so val and test numbers are
    comparable. Every dropped item (hallucinated text, bad/empty type, non-object)
    is something the model asserted that is not a valid gold entity, so it must
    count against precision. Each sentinel gets unique negative offsets: gold spans
    are always >= 0, so a sentinel can never equal a gold key (exact) nor overlap a
    gold span (relaxed) — it only raises the precision denominator."""
    spans = []
    for i, d in enumerate(dropped):
        start = -2 * (i + 1)
        etype = d.get("type") if d.get("type") in ENTITY_TYPES else "_invalid"
        spans.append({"type": etype, "start": start, "end": start + 1})
    return spans


def score_and_report(out_path: Path, gold_rows: list[dict]) -> dict:
    """Score the written predictions against the gold rows this run covered.
    Rows absent from the output file (an aborted run) score as empty predictions,
    which is reported explicitly as `missing` rather than silently dropped."""
    pred_by_id = {r["id"]: r.get("pred", []) + dropped_as_fp(r.get("dropped", []))
                  for r in load_jsonl(out_path)}
    missing = [r["id"] for r in gold_rows if r["id"] not in pred_by_id]

    gold_sentences = [r["entities"] for r in gold_rows]
    pred_sentences = [pred_by_id.get(r["id"], []) for r in gold_rows]
    scores = score_corpus_both(gold_sentences, pred_sentences)

    ex, rel = scores["exact"]["overall"], scores["relaxed"]["overall"]
    print(f"\n===== TEST SET RESULTS — {out_path.name} =====")
    print(f"scored {len(gold_rows) - len(missing)}/{len(gold_rows)} criteria"
          + (f"  ({len(missing)} missing -> counted as no prediction)" if missing else ""))
    print(f"  STRICT   P={ex['precision']:.3f}  R={ex['recall']:.3f}  F1={ex['f1']:.3f}")
    print(f"  RELAXED  P={rel['precision']:.3f}  R={rel['recall']:.3f}  F1={rel['f1']:.3f}")

    # per-type, both conventions side by side. `support` is the gold count for the
    # type and is identical under strict and relaxed, so it is printed once.
    zero = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    print(f"\nper-type   |{'STRICT':^22}|{'RELAXED':^22}|")
    print(f"{'type':>16}  {'P':>6} {'R':>6} {'F1':>6}  {'P':>6} {'R':>6} {'F1':>6} {'support':>8}")
    for etype, s in sorted(scores["exact"]["per_type"].items(), key=lambda kv: -kv[1]["support"]):
        r = scores["relaxed"]["per_type"].get(etype, zero)
        print(f"{etype:>16}  {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f}  "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {s['support']:>8}")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--n-examples", type=int, default=500, help="first N few-shot examples from the superset (default 500, the best val config; n=0 = zero-shot)")
    ap.add_argument("-m", "--m-samples", type=int, default=None, help="number of test criteria to run (default: full test set, 1163)")
    ap.add_argument("--model", default="gemma4:e4b", help="Ollama model tag (default gemma4:e4b)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy/reproducible (default)")
    ap.add_argument("--num-ctx", type=int, default=65536, help="Ollama context window (default 65536, fits the full superset)")
    ap.add_argument("--num-predict", type=int, default=2048, help="max tokens generated per request; caps runaway generation (default 2048)")
    ap.add_argument("--seed", type=int, default=42, help="seed for Ollama sampling (run-to-run reproducibility) and test shuffle (default 42)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle test before sampling (default: file order, stable across reruns)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent Ollama requests (default 8)")
    ap.add_argument("--score-only", action="store_true", help="skip inference and just score the existing output file for this config")
    args = ap.parse_args()

    examples = load_examples(args.n_examples)
    test = load_test(args.m_samples, args.seed, shuffle=args.shuffle)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model)
    m_label = "full" if args.m_samples is None else str(args.m_samples)
    out_path = OUT_DIR / f"predictions_{model_slug}_test_n{args.n_examples}_m{m_label}_seed{args.seed}.jsonl"

    if args.score_only:
        if not out_path.exists():
            sys.exit(f"ERROR: no predictions to score at {out_path}")
        score_and_report(out_path, test)
        return

    check_ollama()  # fail fast if the server is down, before writing anything

    # Resume: skip criteria already written to this exact output file, append the rest.
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"resuming: {len(done_ids)} rows already in {out_path.name}")

    print(f"SPLIT = test | model = {args.model} | n_examples = {len(examples)} | m_samples = {len(test)} "
          f"| temp = {args.temperature} | num_ctx = {args.num_ctx} | workers = {args.workers} "
          f"| outputs = {out_path}")

    todo = [row for row in test if row["id"] not in done_ids]

    n_ok = n_fail = n_dropped = done = 0
    lock = threading.Lock()
    t0 = time.perf_counter()

    with out_path.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(predict, row, args.model, examples, args.temperature,
                               args.num_ctx, args.seed, args.num_predict): row for row in todo}
        try:
            for fut in as_completed(futures):
                row, pred, dropped, status, ok = fut.result()  # RequestException re-raises here
                with lock:
                    n_dropped += len(dropped)
                    n_ok += ok
                    n_fail += (not ok)
                    done += 1
                    fh.write(json.dumps({
                        "id": row["id"],
                        "nct_id": row["nct_id"],
                        "criteria_type": row["criteria_type"],
                        "text": row["text"],
                        "entities": row["entities"],
                        "pred": pred,
                        "dropped": dropped}, ensure_ascii=False) + "\n")
                    fh.flush()
                    print(f"  [{done}/{len(todo)}] {row['id']}: {status}")
                    for d in dropped:
                        print(f"    unplaceable [{d['reason']}] {d['type']}: {d['text']!r}")
        except requests.RequestException as err:
            for f in futures:
                f.cancel()
            fh.flush()
            sys.exit(f"\nERROR: Ollama request failed ({err}). Re-run the same command to resume from here.")

    sort_output_path(out_path)  # deterministic id order across runs
    elapsed = time.perf_counter() - t0

    print(f"ok={n_ok} fail={n_fail} unplaceable_preds={n_dropped} skipped_done={len(done_ids)} "
          f"| elapsed={elapsed:.1f}s | output saved at -> {out_path}")

    score_and_report(out_path, test)


def sort_output_path(path: Path) -> None:
    """Rewrite the JSONL sorted by id so repeated full runs produce identical files
    (concurrent workers write in completion order, which varies run to run)."""
    rows = load_jsonl(path)
    rows.sort(key=lambda r: r["id"])
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
