"""Load the shared, fixed CHIA split and turn it into model-ready tensors.

Downloads/caches train.jsonl / val.jsonl / test.jsonl / *_spans.jsonl /
label_list.json from the team's shared repo (data/processed_baseline). Nothing
here re-splits or regenerates that data -- it's consumed exactly as given.
"""

import json
from pathlib import Path

import requests
import torch

import config


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def ensure_data(cache_dir: Path = config.DATA_CACHE_DIR) -> dict[str, Path]:
    """Download (if not already cached) every file listed in config.DATA_FILES."""
    paths = {}
    for key, filename in config.DATA_FILES.items():
        url = f"{config.DATA_REPO_RAW_BASE}/{filename}"
        paths[key] = _download(url, cache_dir / filename)
    return paths


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_label_maps(label_list_path: Path) -> tuple[dict[str, int], dict[int, str], list[str]]:
    data = json.loads(label_list_path.read_text(encoding="utf-8"))
    labels = data["labels"]
    label2id = data["label2id"]
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label, labels


def align_labels_with_word_ids(word_ids: list[int | None], ner_tags: list[str], label2id: dict[str, int]) -> list[int]:
    """First subword of each word gets the word's label id; everything else (
    continuation subwords, [CLS]/[SEP]/padding) gets -100 so the loss ignores it.
    """
    aligned = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != prev_word_id:
            aligned.append(label2id[ner_tags[word_id]])
        else:
            aligned.append(-100)
        prev_word_id = word_id
    return aligned


class NERDataset(torch.utils.data.Dataset):
    """Tokenizes CHIA token+BIO records into subword-aligned model inputs."""

    def __init__(self, records: list[dict], tokenizer, label2id: dict[str, int], max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        encoded = self.tokenizer(
            record["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoded.word_ids()
        labels = align_labels_with_word_ids(word_ids, record["ner_tags"], self.label2id)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


def decode_predictions_to_word_tags(word_ids: list[int | None], predicted_ids: list[int], id2label: dict[int, str]) -> list[str]:
    """First-subword-only decoding: one predicted BIO tag per original word.

    Mirrors the convention already used in pubmedbert/baseline.py's predict_batch,
    so training-time monitoring and final evaluation decode predictions the same way.
    """
    num_words = max((w for w in word_ids if w is not None), default=-1) + 1
    word_labels: list[str | None] = [None] * num_words
    for pos, word_id in enumerate(word_ids):
        if word_id is not None and word_labels[word_id] is None:
            word_labels[word_id] = id2label[predicted_ids[pos]]
    return [label if label is not None else "O" for label in word_labels]
