# CHIA preprocessing pipeline

Turns the raw CHIA corpus (BRAT standoff `.txt`/`.ann` files) into split,
model-ready datasets for both the PubMedBERT fine-tuning and GPT-4 prompting
experiments.

## Setup

```bash
cd pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m chia_pipeline.download        # -> ../data/raw/chia_without_scope/*.txt,*.ann
python -m chia_pipeline.build_dataset    # -> ../data/processed/{train,val,test}[_spans].jsonl
python -m chia_pipeline.stats            # entity counts per split, for sanity-checking / the paper
python -m chia_pipeline.eda              # sentence-length/entity-density/OOV stats
python -m chia_pipeline.baselines        # all-O and lookup-baseline entity-level P/R/F1
python -m pytest                         # unit tests for the parser/aligner/scorer
```

## Output (`data/processed/`)

- `{split}.jsonl` — `{id, nct_id, criteria_type, tokens, ner_tags}`, BIO-tagged,
  ready for `datasets.load_dataset("json", ...)` + HuggingFace token classification.
- `{split}_spans.jsonl` — `{id, nct_id, criteria_type, text, entities}`, raw
  sentence text with character-offset gold entities. Use this for GPT-4
  few-shot examples and for scoring GPT-4's output.
- `label_list.json` — the BIO label vocabulary (`label2id`/id order).

Both formats are decoded from the *same* flattened gold entities, so
`chia_pipeline.eval_utils.score_corpus` scores PubMedBERT and GPT-4 against
identical gold spans (required for a fair comparison in the paper).

Splits are 80/10/10 train/val/test at the **trial (NCT ID) level** (fixed
seed=42), so inclusion/exclusion sentences from the same trial never land in
different splits.

## File Guide

### `pipeline/src/chia_pipeline/` (the package)

| File | Purpose |
|---|---|
| `__init__.py` | Empty; marks `chia_pipeline` as an importable package. |
| `download.py` | Downloads and unzips the raw CHIA corpus from figshare into `data/raw/chia_without_scope/`. |
| `brat.py` | Parses `.ann` BRAT files into `Fragment` (char-offset entity mention) objects, splitting discontinuous mentions and dropping non-kept types. |
| `constants.py` | Defines kept entity types, the derived BIO label list, and train/val/test split fractions/seed. |
| `tokenizer.py` | Lightweight regex tokenizer producing word tokens with character offsets. |
| `align.py` | Resolves overlapping entity fragments (keep-longest) and converts between token-level BIO tags and char-offset entity spans. |
| `build_dataset.py` | Main pipeline: reads raw docs, aligns entities to sentences/tokens, splits by trial ID, writes `data/processed/*.jsonl` + `label_list.json`. |
| `stats.py` | Prints per-split entity-type and document/sentence counts from the processed data. |
| `eda.py` | Computes exploratory stats (sentence length, entity density, OOV rate) per split. |
| `eval_utils.py` | Shared entity-level precision/recall/F1 scorer used by every baseline/model. |
| `baselines.py` | Trains/evaluates the non-neural all-O and lookup baselines against the processed data. |

### `pipeline/tests/`

| File | Purpose |
|---|---|
| `test_brat.py` | Checks `.ann` T-line parsing produces correct `Fragment`s. |
| `test_align.py` | Checks overlap resolution and BIO↔span conversion round-trip correctly. |
| `test_eval_utils.py` | Checks `score_corpus` gives correct precision/recall/F1 on known gold/pred pairs. |


## Design notes / known limitations

- **Data source**: `chia_without_scope`, not `chia_with_scope`. The with-scope
  release adds a `Scope` annotation layer (coordination/relation boundaries)
  on top of the original 15-type CHIA schema. `Scope` is almost always the
  longest span in an overlap group, so under keep-longest BIO flattening it
  systematically evicted the clinically meaningful entities nested inside it
  — 14.5k of 15.9k dropped mentions were dropped *because a Scope span won*,
  including 5.2k `Condition` and 2.4k `Drug` mentions. Switching sources (no
  code change to the resolution logic itself) cut overlap loss from ~40% to
  ~8.5% and matches the schema prior CHIA NER baselines (e.g. Gao et al.
  2022) used, making results comparable.
- **Entity types**: only the 16 types in `constants.KEPT_ENTITY_TYPES`
  (CHIA's "CONCEPTS" + "ANNOTATION" groups, minus `Scope`) are kept. CHIA
  also has an "ERROR" category (`Non-representable`, `Parsing_Error`, etc.)
  marking annotator data-quality issues, not real entities — those spans are
  dropped. This uses an allowlist, so any other stray tag in the raw data is
  dropped too.
- **Discontinuous entities** (~7% of mentions, e.g. "major impairment of
  renal *[or hepatic]* function" tagged as one entity with a gap) are split
  into independent same-type fragments, since flat BIO can't represent gaps
  within one entity.
- **Overlapping/nested entities** (a small residual amount remains even
  without Scope, e.g. a longer `Temporal` or `Condition` span occasionally
  containing a shorter one) are resolved by keeping the longest span and
  dropping shorter overlaps, since BIO is a flat scheme. Now only ~8.5% of
  mentions, with no single dominant cause. The raw `.ann` files still have
  the full nested/relational structure if a future pass wants it (e.g. for
  a relation-extraction task).
- Sentence count matches the CHIA paper's reported 12,409 criteria exactly,
  confirming the line-based sentence splitting is correct.
- **OOV / long-sentence risk, checked against PubMedBERT's real tokenizer**
  (`reports/tokenizer_diagnostics.py`, needs `pip install -e ".[report]"`):
  true `[UNK]` rate is 0.0% on every split (naive whole-word vocabulary
  overlap overstates this for a subword model), and only ~0.1-0.25% of
  sentences exceed 128 subword tokens — standard `truncation=True,
  max_length=128` is sufficient for fine-tuning, no special handling needed.
