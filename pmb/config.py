"""Fixed settings for the PubMedBERT/BiomedBERT fine-tuning track.

Hyperparameters match Li et al. 2022's tenfold-CV setup (Table 3), since that's
the comparative NER study this track's numbers are meant to be measured against.
"""

from pathlib import Path

MODEL_CHECKPOINT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# Shared, fixed data split -- do not re-split, do not regenerate from raw CHIA.
# All tracks (PubMedBERT, GPT-4 zero/few-shot) score against this same data.
DATA_REPO_RAW_BASE = (
    "https://raw.githubusercontent.com/jatinpsingh/NER_Clinical_Trial_Eligibility"
    "/main/data/processed_baseline"
)
DATA_FILES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
    "train_spans": "train_spans.jsonl",
    "val_spans": "val_spans.jsonl",
    "test_spans": "test_spans.jsonl",
    "label_list": "label_list.json",
    "fold_assignments": "fold_assignments.json",
}

PMB_ROOT = Path(__file__).resolve().parent
DATA_CACHE_DIR = PMB_ROOT / "data_cache"
OUTPUT_DIR = PMB_ROOT / "outputs"
RESULTS_PATH = PMB_ROOT / "results.json"

# 10-fold CV (Li et al. 2022's protocol). The shared repo only materializes one
# fold (test=0, val=1) as train/val/test.jsonl -- since those three files union
# to the complete 12,409-sentence corpus, every other fold's split is
# reconstructed locally from fold_assignments.json's trial->fold mapping
# (verified byte-for-byte identical to the shipped fold-0 files before use).
N_FOLDS = 10
CV_OUTPUT_DIR = PMB_ROOT / "outputs" / "cv"
CV_RESULTS_PATH = PMB_ROOT / "cv_results.json"

# Li et al. 2022, Table 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
NUM_EPOCHS = 10
MAX_SEQ_LENGTH = 256
ADAM_EPSILON = 1e-8

SEED = 42
