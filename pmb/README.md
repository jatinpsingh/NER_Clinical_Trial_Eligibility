# PubMedBERT/BiomedBERT fine-tuning track

Fine-tunes `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (the
renamed PubMedBERT-base) on the CHIA eligibility-criteria NER task, using the
team's **shared, fixed data split** so results are directly comparable across
tracks (this PubMedBERT track, GPT-4 zero-shot, GPT-4 few-shot).

Hyperparameters and metrics follow Li et al. 2022 (the comparative NER study
this project's proposal cites as its primary baseline), not this project's own
proposal defaults, so results are directly comparable to that paper's reported
numbers.

**Trained checkpoint**: https://huggingface.co/ptanwar/pubmedbert-chia-ner
(too large for git -- `outputs/full_run/final/` is gitignored; load it directly
with `AutoModelForTokenClassification.from_pretrained("ptanwar/pubmedbert-chia-ner")`).

## Setup

```bash
cd pmb
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs `chia_pipeline` directly from the team's shared
repo (`jatinpsingh/NER_Clinical_Trial_Eligibility`, `pipeline/` subdirectory) --
this is what guarantees the tokenizer and entity-level scorer here exactly
match how the shared data was built, rather than trusting a possibly-diverged
local copy.

## Data

`data.py` downloads (and caches under `data_cache/`, gitignored) the shared,
fixed split directly from
`jatinpsingh/NER_Clinical_Trial_Eligibility/data/processed_baseline/`:
`train.jsonl` / `val.jsonl` / `test.jsonl` (tokens + BIO tags, for training),
their `*_spans.jsonl` counterparts (text + gold char-offset entities, for
scoring), and `label_list.json` (31 labels: `O` + 15 entity types x B/I --
one fewer type than this repo's own local pipeline, which still keeps `Line`).

**Nothing here re-splits or regenerates this data.** It's the one fixed fold
(`test_fold=0, val_fold=1` per the shared repo's `fold_assignments.json`) every
track is meant to score against.

## Usage

```bash
python train.py --smoke-test      # ~40 examples, 1 epoch -- verifies the full
                                   # pipeline runs and times a step before
                                   # committing to the full job
python train.py                   # full run: lr=5e-5, batch=8, 10 epochs,
                                   # max_seq_len=256, Adam eps=1e-8 (Li et al.
                                   # 2022, Table 3)

python evaluate.py --checkpoint outputs/full_run/final --split both
```

`evaluate.py` reports **strict** entity-level P/R/F1 (exact `(type, start,
end)` match) and **relaxed** (correct type + overlapping span) via the shared
`chia_pipeline.eval_utils.score_corpus` / `score_corpus_relaxed` (Jay's PR #2).
It also writes `outputs/full_run/{split}_predictions.jsonl` in the `{"id",
"entities"}` format `chia_pipeline.evaluate` (the shared CLI) expects, so
results here can be independently re-verified with the team's standard tool:

```bash
python -m chia_pipeline.evaluate --gold data_cache/test_spans.jsonl \
    --pred outputs/full_run/test_predictions.jsonl --pred-format spans
```

Both paths were cross-checked against each other on this track's actual
predictions and produced identical numbers to 9+ decimal places (see
`outputs/full_run/official_report_{val,test}.json`).

## 10-fold cross-validation

`run_cv.py` reconstructs and runs all 10 of the shared repo's CV folds (Li et
al. 2022's protocol), not just the one materialized fold above. Since
`train.jsonl` + `val.jsonl` + `test.jsonl` union to the complete 12,409-sentence
corpus, every other fold's split is rebuilt locally from
`fold_assignments.json`'s trial-to-fold mapping (`data.build_fold_split`) --
verified byte-for-byte identical to the shipped fold-0 files before being
trusted for the other 9.

```bash
python run_cv.py                  # all 10 folds (~2 hours on a single GPU)
python run_cv.py --folds 0 1 2    # a subset, e.g. to resume or split across machines
```

Each fold trains fresh (same hyperparameters/procedure as `train.py`), scores
against that fold's own held-out test set, and its checkpoint is deleted right
after scoring -- only the small `cv_results.json` summary is kept, so disk
usage stays flat across all 10 folds rather than accumulating ~4GB of
checkpoints. Results (mean +/- std across folds, `cv_results.json`):

| | Precision | Recall | F1 |
|---|---|---|---|
| Test, strict | 0.657 +/- 0.013 | 0.682 +/- 0.021 | **0.669 +/- 0.013** |
| Test, relaxed | 0.758 +/- 0.014 | 0.787 +/- 0.024 | **0.772 +/- 0.015** |

The 10-fold mean (0.669 strict) is higher than both the single-fold result
above (0.656) and Li et al.'s published number (0.622) -- the original
fold-0 test set happened to be on the harder end of the 10, not
representative of the average. `outputs/cv_output.log` has the per-fold
breakdown.

## Architecture notes

- `data.py` -- downloads/caches the shared split, builds `label2id`/`id2label`
  from the shared `label_list.json`, aligns BIO labels to PubMedBERT's subword
  tokens (first subword of each word gets the gold label; continuation
  subwords and specials get `-100`, ignored by the loss).
- `metrics.py` -- a small, self-contained, word-index entity-level F1 used
  *only* for per-epoch validation monitoring during training (to pick the best
  checkpoint) -- not the authoritative reported score. It doesn't need
  character offsets or `chia_pipeline` at all, since word-index spans and
  char-offset spans are equivalent coordinate systems for exact-match scoring
  on the same tokenization.
- `train.py` -- loads the checkpoint via `AutoModelForTokenClassification`
  (this discards the pretrained masked-language-model head and attaches a
  fresh, randomly-initialized `768 -> 31` classification layer), fine-tunes
  with HuggingFace `Trainer`.
- `evaluate.py` -- the authoritative, reported score: retokenizes each
  `*_spans.jsonl` sentence's raw text with `chia_pipeline.tokenizer.tokenize`,
  asserts it matches the stored `tokens` (guards against the two data formats
  silently drifting apart), decodes model predictions back into entity spans
  via `chia_pipeline.align.bio_to_spans`, and scores against gold with
  `chia_pipeline.eval_utils`.

## Known limitations

- The full 10-fold distribution (`cv_results.json`) still can't be used for a
  valid Wilcoxon test *against Li et al.'s* number specifically -- they only
  published their mean, not their 10 raw per-fold scores. It *can* validly
  compare this track's own fold-wise distribution against another track's
  repeated-run numbers (e.g. the SLM/LLM tracks), if those exist.
