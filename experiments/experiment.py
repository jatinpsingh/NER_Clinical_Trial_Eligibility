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

ROOT = Path("/home/jatin/nlp/project")
SUPERSET = ROOT / "experiments" / "example_superset_seed42.json"
VAL_SPANS = ROOT / "data" / "processed_baseline" / "val_spans.jsonl"
OUT_DIR = ROOT / "experiments" / "outputs"

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_URL = OLLAMA_HOST + "/api/chat"

ENTITY_TYPES = [
    "Person", "Condition", "Drug", "Observation", "Measurement", "Procedure",
    "Device", "Visit", "Negation", "Qualifier", "Temporal", "Value",
    "Multiplier", "Reference_point", "Mood",
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                    "text": {"type": "string"},
                },
                "required": ["type", "text"],
            },
        }
    },
    "required": ["entities"],
}

# One-line definition per entity class. Definitions (not just the type list) were
# the single biggest lever on the frontier arm (+0.126 strict F1 at 0-shot in
# notebooks/llm.ipynb), so we bake them into the system prompt. Keyed and ordered
# to match ENTITY_TYPES exactly — no `Line` (not in our 15-type schema).
GUIDELINES = {
    "Person": "demographic information describing a person: age, gender, race, ethnicity, etc.",
    "Condition": "a disease or medical condition stated as a diagnosis, sign, or symptom, observed by a provider or reported by the patient",
    "Drug": "a biochemical substance administered to exert a physiological effect; includes prescription and OTC medicines, vaccines, biologics",
    "Observation": "a clinical fact about a person not covered by other domains (e.g. social/lifestyle facts, medical history, family history)",
    "Measurement": "a structured value (numerical or categorical) obtained through standardized examination or testing (e.g. lab tests, vital signs)",
    "Procedure": "an activity or process ordered or carried out by a provider for a diagnostic or therapeutic purpose",
    "Device": "a physical object or instrument used for diagnostic or therapeutic purposes (e.g. pacemakers, stents, syringes, sutures)",
    "Visit": "the location or setting where a person receives medical services (e.g. outpatient, inpatient, emergency room, long-term care)",
    "Negation": "a cue that provokes a Boolean negation on its parent entity (e.g. 'no', 'not', 'without')",
    "Qualifier": "text that further constrains its parent, e.g. location ('facial' trauma), severity ('severe' impairment), or type ('familial' diabetes)",
    "Temporal": "a time expression, duration, or frequency constraining an entity",
    "Value": "a numeric value or range attached to a Measurement (e.g. '<8 g/dL')",
    "Multiplier": "specifies dosage of a Drug or repetition of an entity (e.g. 'at least two of...')",
    "Reference_point": "a concept whose timestamp anchors a parent Temporal; e.g. in 'within two weeks of a blood transfusion', 'blood transfusion'",
    "Mood": "text that transforms its parent into a non-literal statement; e.g. in 'eligible for surgery', 'eligible for'",
}

SYSTEM = (
    "You are a clinical NLP annotator. Extract named entities from a single clinical "
    "trial eligibility criterion. Use ONLY these entity classes (definitions follow):\n"
    + "\n".join(f"- {t}: {GUIDELINES[t]}" for t in ENTITY_TYPES) + "\n"
    "Each entity's text must be copied verbatim from the criterion — do not paraphrase, "
    "normalize, expand abbreviations, or invent text. Return every entity you find. "
    "If none, return an empty list."
)

def load_examples(n: int) -> list[dict]:
    data = json.loads(SUPERSET.read_text())
    ordered = [data[str(i)] for i in range(len(data))]
    return ordered[:n]

def load_val(m: int | None, seed: int, shuffle: bool) -> list[dict]:
    rows = [json.loads(l) for l in VAL_SPANS.open()]
    if shuffle:
        random.Random(seed).shuffle(rows)
    return rows if m is None else rows[:m]

def build_messages(examples: list[dict], text: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in examples:
        ents = [{"type": e["type"], "text": e["text"]} for e in ex["entities"]]
        msgs.append({"role": "user", "content": ex["text"]})
        msgs.append({"role": "assistant",
                     "content": json.dumps({"entities": ents}, ensure_ascii=False)})
    msgs.append({"role": "user", "content": text})
    return msgs

def call_ollama(model: str, messages: list[dict], temperature: float, num_ctx: int, seed: int) -> dict:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": RESPONSE_SCHEMA,
            # seed is required for run-to-run reproducibility: temperature=0 alone is
            # NOT deterministic on Ollama/llama.cpp (batched-reduction float ordering).
            "options": {"temperature": temperature, "num_ctx": num_ctx, "seed": seed},
        },
        timeout=300,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    return json.loads(content)

def sort_output(path: Path) -> None:
    """Rewrite the JSONL sorted by id so repeated full runs produce identical files
    (concurrent workers write in completion order, which varies run to run)."""
    rows = [json.loads(l) for l in path.open()]
    rows.sort(key=lambda r: r["id"])
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def check_ollama() -> None:
    try:
        requests.get(OLLAMA_HOST, timeout=5).raise_for_status()
    except requests.RequestException as err:
        sys.exit(f"ERROR: cannot reach Ollama at {OLLAMA_HOST} ({err}). Is `ollama serve` running?")

def predict(row: dict, model: str, examples: list[dict], temperature: float,
            num_ctx: int, seed: int) -> tuple[dict, list[dict], list[dict], str, bool]:
    """Runs in a worker thread. Only requests.RequestException propagates (a
    server-level failure aborts the run so it can be resumed); ANY malformed
    model output for this one row is caught and recorded as an empty prediction
    so a single bad response never kills the whole batch."""
    try:
        raw = call_ollama(model, build_messages(examples, row["text"]), temperature, num_ctx, seed)
        entities = raw.get("entities", []) if isinstance(raw, dict) else []
        pred, dropped = locate(row["text"], entities)
        status = f"{len(pred)} found" + (f" ({len(dropped)} unplaceable)" if dropped else "")
        return row, pred, dropped, status, True
    except requests.RequestException:
        raise  # server down / connection lost -> abort so the run can resume
    except Exception as err:  # bad JSON, non-dict output, etc. -> just this row fails
        return row, [], [], f"FAIL ({type(err).__name__}: {err})", False


def locate(text: str, entities: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (located_spans, dropped) where each dropped item is the raw model
    entity we could not place, tagged with a `reason` for error analysis."""
    lower_text = text.lower()
    out, seen, cursor, dropped = [], set(), 0, []
    for e in entities:
        if not isinstance(e, dict):  # model emitted a bare string / non-object item
            dropped.append({"type": "", "text": repr(e), "reason": "not_an_object"})
            continue
        span = (e.get("text") or "").strip()
        etype = e.get("type", "")
        if not span or etype not in ENTITY_TYPES:
            dropped.append({"type": etype, "text": e.get("text", ""), "reason": "empty_or_bad_type"})
            continue
        needle = span.lower()
        idx = lower_text.find(needle, cursor)
        if idx == -1:
            idx = lower_text.find(needle)  # span out of the model's claimed order
        if idx == -1:
            # not verbatim even case-insensitively — hallucinated/paraphrased
            dropped.append({"type": etype, "text": span, "reason": "not_in_text"})
            continue
        end = idx + len(needle)
        key = (etype, idx, end)
        if key in seen:
            continue  # exact duplicate prediction
        seen.add(key)
        out.append({"type": etype, "start": idx, "end": end, "text": text[idx:end]})
        cursor = end
    return out, dropped

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--n-examples", type=int, default=10, help="first N few-shot examples from the superset (default 10; n=0 = zero-shot)")
    ap.add_argument("-m", "--m-samples", type=int, default=None, help="number of val criteria to run (default: full val set)")
    ap.add_argument("--model", default="gemma4:e2b", help="Ollama model tag")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy/reproducible (default)")
    ap.add_argument("--num-ctx", type=int, default=65536, help="Ollama context window (default 65536, fits the full 500-shot superset)")
    ap.add_argument("--seed", type=int, default=42, help="seed for Ollama sampling (run-to-run reproducibility) and val shuffle (default 42)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle val before sampling (default: file order, stable across reruns)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent Ollama requests (default 4)")
    args = ap.parse_args()

    check_ollama()  # fail fast if the server is down, before writing anything

    examples = load_examples(args.n_examples)
    val = load_val(args.m_samples, args.seed, shuffle=args.shuffle)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model)
    m_label = "full" if args.m_samples is None else str(args.m_samples)
    out_path = OUT_DIR / (f"predictions_{model_slug}_n{args.n_examples}_m{m_label}_seed{args.seed}.jsonl")

    # Resume: skip criteria already written to this exact output file, append the rest.
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"resuming: {len(done_ids)} rows already in {out_path.name}")

    print(f"model = {args.model} | n_examples = {len(examples)} | m_samples = {len(val)} "
          f"| temp = {args.temperature} | num_ctx = {args.num_ctx} | workers = {args.workers} "
          f"| outputs = {out_path}")

    todo = [row for row in val if row["id"] not in done_ids]

    n_ok = n_fail = n_dropped = done = 0
    lock = threading.Lock()
    t0 = time.perf_counter()

    with out_path.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(predict, row, args.model, examples, args.temperature,
                               args.num_ctx, args.seed): row for row in todo}
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

    sort_output(out_path)  # deterministic id order across runs
    elapsed = time.perf_counter() - t0

    print(f"ok={n_ok} fail={n_fail} unplaceable_preds={n_dropped} skipped_done={len(done_ids)} | elapsed={elapsed:.1f}s | output saved at -> {out_path}")


if __name__ == "__main__":
    main()