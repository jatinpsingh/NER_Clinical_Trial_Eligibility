from pathlib import Path
import json
import random

ROOT = Path("/home/jatin/nlp/project")
DATA = ROOT / "data" / "processed_baseline"
TRAIN_SPANS = DATA / "train_spans.jsonl"
VAL_SPANS   = DATA / "val_spans.jsonl"

SEED = 42

train = [json.loads(l) for l in open(TRAIN_SPANS)]
val = [json.loads(l) for l in open(VAL_SPANS)]

N = 1000  # superset = largest few-shot size you'll ever sample from this file

OUT = ROOT / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

ENTITY_TYPES = ["Person", "Condition", "Drug", "Observation", "Measurement",
                "Procedure", "Device", "Visit", "Negation", "Qualifier",
                "Temporal", "Value", "Multiplier", "Reference_point", "Mood"]


def valid_example(r):
    # clean, prompt-efficient few-shot demos: offset-valid, not too short/long, 1-6 entities
    return (1 <= len(r["entities"]) <= 6 and 20 <= len(r["text"]) <= 200
            and all(e["start"] is not None
                    and r["text"][e["start"]:e["end"]] == e["text"]
                    for e in r["entities"]))


random.Random(SEED).shuffle(train)

samples = []
covered = set()
pool = [r for r in train if valid_example(r)]   # filter first, then greedy

# greedy: cover all 15 types with as few examples as possible...
while pool and len(covered) < len(ENTITY_TYPES):
    best, gain = None, 0
    for trail in pool:
        n_type = len({e["type"] for e in trail["entities"]} - covered)
        if n_type > gain:
            gain, best = n_type, trail
    if best is None:
        break
    samples.append(best)
    pool.remove(best)
    covered.update(e["type"] for e in best["entities"])

# ...then fill the rest in seeded-shuffled order, truncate to N
samples = (samples + pool[: max(0, N - len(samples))])[:N]

assert len(covered) == len(ENTITY_TYPES), f"only covered {len(covered)}/15 types"
print(f"saved {len(samples)} examples | types covered {len(covered)}/15")

sample_set = OUT / f"example_superset_seed{SEED}.json"
sample_set.write_text(json.dumps({str(i): s for i, s in enumerate(samples)}, ensure_ascii=False))
print(f"wrote -> {sample_set}")
