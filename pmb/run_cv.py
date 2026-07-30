"""Run the full 10-fold cross-validation (Li et al. 2022's protocol).

For each fold: reconstruct that fold's train/val/test split locally (see
data.build_fold_split), fine-tune fresh (same hyperparameters/procedure as
the single-fold run), evaluate on that fold's held-out test set, and record
strict + relaxed entity-level P/R/F1. After all 10 folds, report mean +
standard deviation -- the number actually comparable to Li et al.'s own
10-fold-averaged result, instead of a single run.

Optimized to avoid unnecessary disk/compute overhead across folds:
  - The tokenizer and pretrained-checkpoint download are only fetched once
    (HuggingFace caches the download locally; re-instantiating the model
    per fold with a fresh random classifier head still reads from that
    local cache, no re-download).
  - Evaluation runs against the in-memory model straight after training --
    no save-to-disk-then-reload round trip.
  - Per-epoch checkpoints (needed only for load_best_model_at_end) are
    capped at save_total_limit=1 during training, then the whole per-fold
    checkpoint directory is deleted once the in-memory model has been
    scored -- keeping disk usage roughly constant across all 10 folds
    instead of accumulating ~4GB of checkpoints we don't need to keep.

Usage:
    python run_cv.py                  # all 10 folds
    python run_cv.py --folds 0 1 2    # a subset (e.g. to resume/parallelize)
"""

import argparse
import json
import shutil
import statistics
import sys
import time

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import config
from data import NERDataset, build_fold_split, ensure_data, load_label_maps
from evaluate import get_device, score_records
from metrics import word_level_entity_f1
from train import make_compute_metrics, set_seed


def run_one_fold(fold: int, tokenizer, label2id, id2label, labels, device, smoke_test: bool = False) -> dict:
    print(f"\n=== Fold {fold}/{config.N_FOLDS - 1} ===", file=sys.stderr)
    set_seed(config.SEED)

    paths = ensure_data()
    token_splits, span_splits = build_fold_split(paths, fold)
    if smoke_test:
        for splits in (token_splits, span_splits):
            splits["train"] = splits["train"][:40]
            splits["val"] = splits["val"][:10]
            splits["test"] = splits["test"][:10]
    print(
        f"  train={len(token_splits['train'])} val={len(token_splits['val'])} "
        f"test={len(token_splits['test'])}",
        file=sys.stderr,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_CHECKPOINT, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    train_dataset = NERDataset(token_splits["train"], tokenizer, label2id, config.MAX_SEQ_LENGTH)
    val_dataset = NERDataset(token_splits["val"], tokenizer, label2id, config.MAX_SEQ_LENGTH)
    collator = DataCollatorForTokenClassification(tokenizer)

    fold_dir = config.CV_OUTPUT_DIR / f"fold_{fold}"
    training_args = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=1 if smoke_test else config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        adam_epsilon=config.ADAM_EPSILON,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # keep disk bounded -- only need the model in memory after
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=config.SEED,
        logging_strategy="epoch",
        report_to=[],
        dataloader_pin_memory=False,
        # No multiprocessing dataloader workers here (unlike train.py's single-
        # Trainer run): reusing one tokenizer across repeated Trainer/fold
        # iterations in one long-running process hit a fork-related hang with
        # HuggingFace's fast tokenizer + persistent workers on macOS. Not worth
        # the risk in a 10-fold loop -- single-threaded loading is fast enough
        # given the model-compute time dominates regardless.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(id2label),
        processing_class=tokenizer,
    )

    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start
    print(f"  training took {elapsed:.1f}s", file=sys.stderr)

    model = trainer.model.to(device).eval()

    val_scores, _ = score_records(token_splits["val"], span_splits["val"], model, tokenizer, device)
    test_scores, _ = score_records(token_splits["test"], span_splits["test"], model, tokenizer, device)
    print(
        f"  val strict F1={val_scores['strict']['overall']['f1']:.4f}  "
        f"test strict F1={test_scores['strict']['overall']['f1']:.4f}  "
        f"test relaxed F1={test_scores['relaxed']['overall']['f1']:.4f}",
        file=sys.stderr,
    )

    # Free the checkpoint directory -- only the small JSON summary is kept.
    del trainer, model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(fold_dir, ignore_errors=True)

    return {
        "fold": fold,
        "num_train": len(token_splits["train"]),
        "num_val": len(token_splits["val"]),
        "num_test": len(token_splits["test"]),
        "elapsed_seconds": elapsed,
        "train_loss": train_result.metrics.get("train_loss"),
        "val": val_scores,
        "test": test_scores,
    }


def summarize(fold_results: list[dict]) -> dict:
    def stats(values: list[float]) -> dict:
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "values": values,
        }

    summary = {}
    for split in ("val", "test"):
        summary[split] = {}
        for mode in ("strict", "relaxed"):
            for metric in ("precision", "recall", "f1"):
                values = [r[split][mode]["overall"][metric] for r in fold_results]
                summary[split].setdefault(mode, {})[metric] = stats(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(config.N_FOLDS)))
    parser.add_argument("--smoke-test", action="store_true", help="tiny slice, 1 epoch, quick pipeline check")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}", file=sys.stderr)

    paths = ensure_data()
    label2id, id2label, labels = load_label_maps(paths["label_list"])
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)

    config.CV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_results = []
    overall_start = time.time()
    for fold in args.folds:
        result = run_one_fold(fold, tokenizer, label2id, id2label, labels, device, smoke_test=args.smoke_test)
        fold_results.append(result)
        # Write incrementally so partial progress survives an interruption.
        config.CV_RESULTS_PATH.write_text(
            json.dumps({"folds": fold_results, "summary": summarize(fold_results)}, indent=2)
        )

    total_elapsed = time.time() - overall_start
    summary = summarize(fold_results)
    print(f"\n=== Done: {len(fold_results)} folds in {total_elapsed:.1f}s ===", file=sys.stderr)
    print(
        f"Test strict F1:  {summary['test']['strict']['f1']['mean']:.4f} "
        f"+/- {summary['test']['strict']['f1']['std']:.4f}",
        file=sys.stderr,
    )
    print(
        f"Test relaxed F1: {summary['test']['relaxed']['f1']['mean']:.4f} "
        f"+/- {summary['test']['relaxed']['f1']['std']:.4f}",
        file=sys.stderr,
    )
    print(f"Saved full results to {config.CV_RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
