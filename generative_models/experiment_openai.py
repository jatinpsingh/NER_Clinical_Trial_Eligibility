#!/usr/bin/env python3
"""
This script runs CHIA NER extraction with an OpenAI API model.

Results are directly comparable with Ollama: identical SYSTEM prompt, identical few-shot superset, 
identical chat message structure, identical locate() logic, identical output records.

Differences from the Ollama script, forced by the API:
  - temperature is NOT sent (this model rejects it); determinism relies on seed
  - num_ctx does not apply
  - structured output uses response_format=json_schema instead of Ollama's format
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

# Paths
ROOT = Path(os.environ.get("CHIA_ROOT", "/content/NER_repo"))
SUPERSET = ROOT / "slm" / "example_superset_seed42.json"
DATA_DIR = ROOT / "data" / "processed_baseline"
OUT_DIR = OUT_DIR = ROOT / "slm" / "outputs" / re.sub(r"[^A-Za-z0-9]+", "-", "gpt-5.6-sol")


# Entity schema
DOMAIN_TYPES = [
    "Person", "Condition", "Drug", "Observation", "Measurement", "Procedure",
    "Device", "Visit",
]
FIELD_TYPES = ["Temporal", "Value"]
CONSTRUCT_TYPES = ["Negation", "Qualifier", "Multiplier", "Reference_point", "Mood"]
ENTITY_TYPES = DOMAIN_TYPES + FIELD_TYPES + CONSTRUCT_TYPES

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
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}

GUIDELINES = {
    "Person": "demographic information used to describe a Person, including age, gender, race, ethnicity, etc.",
    "Condition": "the presence of a disease or medical condition stated as a diagnosis, a sign, or a symptom, which is either observed by a Provider or reported by the patient.",
    "Drug": "a biochemical substance formulated in such a way that when administered to a Person it will exert a certain physiological effect; includes prescription and over-the-counter medicines, vaccines, and large-molecule biologic therapies.",
    "Observation": "clinical facts about a Person obtained in the context of examination, questioning or a procedure; includes any data that cannot be represented by any other domains, such as social and lifestyle facts, medical history, family history, etc.",
    "Measurement": "structured values (numerical or categorical) obtained through systematic and standardized examination or testing of a Person or Person's sample.",
    "Procedure": "activities or processes ordered by, or carried out by, a healthcare provider on the patient to have a diagnostic or therapeutic purpose.",
    "Device": "exposure to a foreign physical object or instrument which is used for diagnostic or therapeutic purposes through a mechanism beyond chemical action; includes implantable objects (e.g. pacemakers, stents, artificial joints), medical equipment and supplies (e.g. bandages, crutches, syringes), other instruments used in medical procedures (e.g. sutures, defibrillators) and material used in clinical care (e.g. adhesives, body material, dental material, surgical material).",
    "Visit": "location or setting in which a Person is receiving medical services from one or more providers, including outpatient care, inpatient confinement, emergency room, and long-term care.",
    "Temporal": "represents a point in the line of time. Most often, a Temporal overlaps a Reference_point entity, and is linked to it via a has_index-type relationship (see definition of Reference_point below).",
    "Value": "represents a structured value, either as a number (e.g. blood pressure < 140/90 mmHg) or as a concept (e.g. elevated serum creatinine). When specifying a number value, the only components accepted inside its free text are (extending the above example): logical operator (<), numeral (140/90), unit of measure (mmHg).",
    "Negation": "provokes a Boolean negation on its parent entity. If the truth value of the parent evaluates to false, it then becomes true, and vice-versa.",
    "Qualifier": "subsets the meaning of its parent by imposing a further constraint. The value of a Qualifier oftentimes serves as a supplement to the value of its parent, that is, it may be the case that the free text contained by a Qualifier can be concatenated with the free text contained by its parent (e.g. a Condition) to form one string that can then be linked to a single code. For example, if the free text reads \"familial diabetes insipidus\", we might have one Condition \"diabetes insipidus\" linked to one Qualifier \"familial.\" Another common case is for Qualifiers to express the anatomic location of a Condition (e.g. facial trauma) or the severity of a Condition (e.g. severe renal impairment).",
    "Multiplier": "specifies either dosage of a Drug entity, or repetition type of entity (e.g. \"at least two of...\").",
    "Reference_point": "Always comes downstream (usually directly) from a parent Temporal, and specifies a concept whose timestamp anchors that Temporal. For example, in \"within two weeks of a blood transfusion\" this entire text string is one Temporal, and it contains (overlaps) the Reference_point \"blood transfusion.\"",
    "Mood": "transforms the meaning of its parent into a different kind of statement that is not about the literal presence of the parent. For example, in \"eligible for surgery\" the Mood \"eligible for\" denotes that satisfying this criterion does not require the presence of records of the surgery, but rather the presence of concept(s) associated to that surgery \u2013 in this case, the patient's eligibility for it.",
}

GROUPS = [
    ("Domain",
     "Domain entities represent semantic categories for a given concept.",
     DOMAIN_TYPES),
    ("Fields",
     "Field entities represent properties of the Domain concepts. They may provide the "
     "value or range of values that must be present in a given lab test or timeframe for a "
     "previous diagnosis, and always appear downstream (even if indirectly) from at least "
     "one Domain entity.",
     FIELD_TYPES),
    ("Constructs",
     "Construct entities serve syntactic purposes. Like Fields, they require a relationship "
     "to another entity to form any meaning. With the exception of Scope (excluded from this "
     "schema), all Construct entities are necessarily children of the entity whose meaning "
     "they complement or modify.",
     CONSTRUCT_TYPES),
]


def _guidelines_block() -> str:
    blocks = []
    for name, intro, types in GROUPS:
        lines = [f"{name}: {intro}"] + [f"  - {t}: {GUIDELINES[t]}" for t in types]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


SYSTEM = (
    "You are a clinical NLP annotator. Extract named entities from a single clinical "
    "trial eligibility criterion. Use ONLY the entity classes below, organized into three "
    "groups (Domain, Fields, Constructs):\n\n"
    + _guidelines_block() + "\n\n"
    "Each entity's text must be copied verbatim from the criterion \u2014 do not paraphrase, "
    "normalize, expand abbreviations, or invent text. Return every entity you find. "
    "If none, return an empty list."
)

# Bare-label system prompt for the no-guidelines ablation (--no-guidelines).
SYSTEM_MINIMAL = (
    "You are a clinical NLP annotator. Extract named entities from a single clinical "
    "trial eligibility criterion. Use ONLY these entity classes: "
    + ", ".join(ENTITY_TYPES) + ".\n\n"
    "Each entity's text must be copied verbatim from the criterion \u2014 do not paraphrase, "
    "normalize, expand abbreviations, or invent text. Return every entity you find. "
    "If none, return an empty list."
)


# Data loading

def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def load_examples(n: int) -> list[dict]:
    data = json.loads(SUPERSET.read_text())
    ordered = [data[str(i)] for i in range(len(data))]
    return ordered[:n]


def load_split(split: str, m: int | None, seed: int, shuffle: bool) -> list[dict]:
    rows = load_jsonl(DATA_DIR / f"{split}_spans.jsonl")
    if shuffle:
        random.Random(seed).shuffle(rows)
    return rows if m is None else rows[:m]


# Prompting

def build_messages(examples: list[dict], text: str, system: str = SYSTEM) -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    for ex in examples:
        ents = [{"type": e["type"], "text": e["text"]} for e in ex["entities"]]
        msgs.append({"role": "user", "content": ex["text"]})
        msgs.append({"role": "assistant",
                     "content": json.dumps({"entities": ents}, ensure_ascii=False)})
    msgs.append({"role": "user", "content": text})
    return msgs


# Span location

def locate(text: str, entities: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (located_spans, dropped); each dropped item carries a reason."""
    lower_text = text.lower()
    out, seen, cursor, dropped = [], set(), 0, []
    for e in entities:
        if not isinstance(e, dict):
            dropped.append({"type": "", "text": repr(e), "reason": "not_an_object"})
            continue
        span = (e.get("text") or "").strip()
        etype = e.get("type", "")
        if not span or etype not in ENTITY_TYPES:
            dropped.append({"type": etype, "text": e.get("text", ""),
                            "reason": "empty_or_bad_type"})
            continue
        needle = span.lower()
        idx = lower_text.find(needle, cursor)
        if idx == -1:
            idx = lower_text.find(needle)
        if idx == -1:
            dropped.append({"type": etype, "text": span, "reason": "not_in_text"})
            continue
        end = idx + len(needle)
        key = (etype, idx, end)
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": etype, "start": idx, "end": end, "text": text[idx:end]})
        cursor = end
    return out, dropped


def parse_reply(raw: str) -> tuple[list[dict], bool]:
    """Model reply string -> (entity dicts, reply_was_parseable)."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return [], False
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return [], False
    if not isinstance(obj, dict):
        return [], False
    ents = obj.get("entities", [])
    return (ents, True) if isinstance(ents, list) else ([], False)



# API call

class Usage:
    """Running token totals for cost estimation."""
    def __init__(self):
        self.prompt = self.cached = self.completion = 0

    def add(self, u):
        self.prompt += u.prompt_tokens
        self.completion += u.completion_tokens
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            self.cached += getattr(details, "cached_tokens", 0) or 0

    def report(self, in_rate, cached_rate, out_rate):
        uncached = max(0, self.prompt - self.cached)
        cost = (uncached * in_rate + self.cached * cached_rate
                + self.completion * out_rate) / 1e6
        return (f"tokens: prompt={self.prompt:,} (cached {self.cached:,}, "
                f"{self.cached / self.prompt:.1%}) completion={self.completion:,} "
                f"| est. cost ${cost:.2f}")


async def call_model(client, model, messages, seed, usage, tries=5):
    for attempt in range(tries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                seed=seed,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "ner", "strict": True,
                                                 "schema": RESPONSE_SCHEMA}},
            )
            if resp.usage:
                usage.add(resp.usage)
            return resp.choices[0].message.content
        except Exception as err:
            msg = str(err)
            if "quota" in msg.lower() or "billing" in msg.lower():
                raise                            # permanent: do not retry
            print(f"    retry {attempt + 1}/{tries}: {type(err).__name__}: {msg[:140]}",
                  file=sys.stderr)
            if attempt == tries - 1:
                raise
            await asyncio.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))



# Output

def sort_output(path: Path) -> None:
    """Rewrite sorted by id, so repeated full runs produce identical files."""
    rows = load_jsonl(path)
    rows.sort(key=lambda r: r["id"])
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")



# Main

async def run(args) -> Path:
    client = AsyncOpenAI()
    system = SYSTEM_MINIMAL if args.no_guidelines else SYSTEM
    examples = load_examples(args.n_examples)
    rows = load_split(args.split, args.m_samples, args.seed, args.shuffle)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model)
    m_label = "full" if args.m_samples is None else str(args.m_samples)
    guide_tag = "" if not args.no_guidelines else "_noguide"
    out_path = OUT_DIR / (f"predictions_{model_slug}_{args.split}_n{args.n_examples}"
                          f"_m{m_label}_seed{args.seed}{guide_tag}.jsonl")

    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"resuming: {len(done_ids)} rows already in {out_path.name}")

    todo = [r for r in rows if r["id"] not in done_ids]

    if args.no_dedup:
        groups = {i: [r] for i, r in enumerate(todo)}
        keys = {i: r["text"] for i, r in enumerate(todo)}
    else:
        groups, keys = {}, {}
        for r in todo:
            groups.setdefault(r["text"], []).append(r)
        keys = {t: t for t in groups}

    print(f"model = {args.model} | n_examples = {len(examples)} | split = {args.split} "
          f"| rows = {len(rows)} | todo = {len(todo)} | api calls = {len(groups)} "
          f"| workers = {args.workers} | seed = {args.seed}")
    print(f"outputs -> {out_path}")

    if not todo:
        print("nothing to do")
        return out_path

    usage = Usage()
    sem = asyncio.Semaphore(args.workers)
    counters = {"done": 0, "fail": 0, "dropped": 0}
    lock = asyncio.Lock()
    t0 = time.perf_counter()

    async def worker(key, fh):
        text = keys[key]
        async with sem:
            raw = await call_model(client, args.model,
                                   build_messages(examples, text, system),
                                   args.seed, usage)
        ents, ok = parse_reply(raw)
        pred, dropped = locate(text, ents)
        async with lock:
            counters["done"] += 1
            counters["fail"] += (not ok)
            counters["dropped"] += len(dropped)
            for r in groups[key]:
                fh.write(json.dumps({
                    "id": r["id"],
                    "nct_id": r.get("nct_id"),
                    "criteria_type": r.get("criteria_type"),
                    "text": r["text"],
                    "entities": r["entities"],
                    "pred": pred,
                    "dropped": dropped}, ensure_ascii=False) + "\n")
            fh.flush()
            if counters["done"] % 50 == 0 or counters["done"] == len(groups):
                print(f"  [{counters['done']}/{len(groups)}] "
                      f"{len(pred)} found"
                      + (f", {len(dropped)} unplaceable" if dropped else ""))

    with out_path.open("a") as fh:
        await asyncio.gather(*(worker(k, fh) for k in groups))

    sort_output(out_path)
    elapsed = time.perf_counter() - t0

    print(f"\ndone in {elapsed:.1f}s | unparseable_replies={counters['fail']} "
          f"| unplaceable_preds={counters['dropped']} | skipped_done={len(done_ids)}")
    if usage.prompt:
        print(usage.report(args.input_rate, args.cached_rate, args.output_rate))
    print(f"saved -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--n-examples", type=int, default=10,
                    help="first N few-shot examples from the superset (default 10; 0 = zero-shot)")
    ap.add_argument("-m", "--m-samples", type=int, default=None,
                    help="number of criteria to run (default: the whole split)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="which split to run on (default val)")
    ap.add_argument("--model", default="gpt-5.6-sol", help="OpenAI model id")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the API and for --shuffle (default 42)")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle the split before sampling (default: file order)")
    ap.add_argument("--workers", type=int, default=32,
                    help="concurrent API requests (default 32)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="send every row separately, even duplicate texts")
    ap.add_argument("--no-guidelines", action="store_true",
                    help="ablation: bare label names instead of the full schema definitions")
    ap.add_argument("--print-prompt", action="store_true",
                    help="print the messages for one criterion and exit (no API calls)")
    ap.add_argument("--input-rate", type=float, default=1.0,
                    help="USD per 1M uncached input tokens, for the cost estimate")
    ap.add_argument("--cached-rate", type=float, default=0.10,
                    help="USD per 1M cached input tokens")
    ap.add_argument("--output-rate", type=float, default=6.0,
                    help="USD per 1M output tokens")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY is not set.")
    if not SUPERSET.exists():
        sys.exit(f"ERROR: superset not found at {SUPERSET} (set CHIA_ROOT?)")
    if not (DATA_DIR / f"{args.split}_spans.jsonl").exists():
        sys.exit(f"ERROR: no {args.split}_spans.jsonl in {DATA_DIR} (set CHIA_ROOT?)")

    if args.print_prompt:
        import hashlib
        system = SYSTEM_MINIMAL if args.no_guidelines else SYSTEM
        print(f"SYSTEM chars={len(system)} "
              f"sha256[:16]={hashlib.sha256(system.encode()).hexdigest()[:16]}\n")
        rows = load_split(args.split, 1, args.seed, args.shuffle)
        for m in build_messages(load_examples(args.n_examples), rows[0]["text"], system):
            print(f"--- {m['role']} ---")
            print(m["content"][:400])
            print()
        return

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
