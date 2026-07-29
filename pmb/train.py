"""Fine-tune PubMedBERT/BiomedBERT on the shared, fixed CHIA split.

Usage:
    python train.py                  # full run: Li et al. 2022 hyperparameters
    python train.py --smoke-test      # tiny slice, 1 epoch -- verifies the
                                      # pipeline runs end-to-end and times a step
                                      # before committing to the full job
"""

import argparse
import json
import random
import sys
import time

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import config
from data import NERDataset, ensure_data, load_jsonl, load_label_maps
from metrics import word_level_entity_f1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_compute_metrics(id2label: dict[int, str]):
    def compute_metrics(eval_pred) -> dict:
        logits, label_ids = eval_pred
        predicted_ids = np.argmax(logits, axis=-1)

        gold_tags_list, pred_tags_list = [], []
        for gold_row, pred_row in zip(label_ids, predicted_ids):
            gold_tags = [id2label[int(g)] for g, p in zip(gold_row, pred_row) if g != -100]
            pred_tags = [id2label[int(p)] for g, p in zip(gold_row, pred_row) if g != -100]
            gold_tags_list.append(gold_tags)
            pred_tags_list.append(pred_tags)

        result = word_level_entity_f1(gold_tags_list, pred_tags_list)
        return {"precision": result["precision"], "recall": result["recall"], "f1": result["f1"]}

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="tiny slice, 1 epoch, quick timing run")
    parser.add_argument("--smoke-test-examples", type=int, default=40)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = get_device()
    print(f"Device: {device}", file=sys.stderr)

    print("Fetching shared, fixed data split...", file=sys.stderr)
    paths = ensure_data()
    label2id, id2label, labels = load_label_maps(paths["label_list"])
    print(f"{len(labels)} labels loaded from shared label_list.json", file=sys.stderr)

    train_records = load_jsonl(paths["train"])
    val_records = load_jsonl(paths["val"])

    if args.smoke_test:
        train_records = train_records[: args.smoke_test_examples]
        val_records = val_records[: max(args.smoke_test_examples // 4, 5)]
        print(
            f"[smoke test] using {len(train_records)} train / {len(val_records)} val examples, 1 epoch",
            file=sys.stderr,
        )

    print(f"Loading tokenizer + model: {config.MODEL_CHECKPOINT}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)
    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_CHECKPOINT,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = NERDataset(train_records, tokenizer, label2id, config.MAX_SEQ_LENGTH)
    val_dataset = NERDataset(val_records, tokenizer, label2id, config.MAX_SEQ_LENGTH)
    collator = DataCollatorForTokenClassification(tokenizer)

    num_epochs = 1 if args.smoke_test else config.NUM_EPOCHS
    output_dir = config.OUTPUT_DIR / ("smoke_test" if args.smoke_test else "full_run")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        adam_epsilon=config.ADAM_EPSILON,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=config.SEED,
        logging_strategy="epoch",
        report_to=[],
        # MPS-specific: pinned memory is a CUDA-only optimization and is a silent
        # no-op (with a warning) on MPS, so turn it off rather than pay the check.
        dataloader_pin_memory=False,
        # Data loading was single-threaded (0 workers) despite 11 CPU cores being
        # available -- overlap tokenization/collation with GPU compute instead.
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        # train_sampling_strategy left at its default ("random"), not
        # "group_by_length": grouping by length is faster (less padding waste)
        # but changes batch composition/order vs. Li et al.'s presumably-random
        # batching. Keeping plain random shuffling here so training dynamics
        # match the paper exactly, at the cost of some wasted padding compute.
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
    print(f"Training took {elapsed:.1f}s for {num_epochs} epoch(s)", file=sys.stderr)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    log = {
        "smoke_test": args.smoke_test,
        "num_train_examples": len(train_records),
        "num_val_examples": len(val_records),
        "num_epochs": num_epochs,
        "elapsed_seconds": elapsed,
        "train_metrics": train_result.metrics,
        "final_checkpoint": str(final_dir),
    }
    log_path = output_dir / "train_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"Saved checkpoint to {final_dir}", file=sys.stderr)
    print(f"Saved training log to {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
