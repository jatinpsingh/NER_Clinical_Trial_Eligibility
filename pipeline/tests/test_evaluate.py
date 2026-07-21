import json

import pytest

from chia_pipeline.evaluate import (
    align_wordpiece_predictions,
    evaluate,
    load_gold,
    load_predictions_spans,
)


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def gold_spans_file(tmp_path):
    records = [
        {
            "id": "s1",
            "nct_id": "NCT1",
            "criteria_type": "inclusion",
            "text": "type 2 diabetic, age 18 and over",
            "entities": [
                {"type": "Condition", "start": 0, "end": 15, "text": "type 2 diabetic"},
                {"type": "Person", "start": 17, "end": 20, "text": "age"},
            ],
        },
        {
            "id": "s2",
            "nct_id": "NCT2",
            "criteria_type": "exclusion",
            "text": "no known drug allergies",
            "entities": [],
        },
    ]
    path = tmp_path / "test_spans.jsonl"
    _write_jsonl(path, records)
    return path


def test_load_gold_indexes_by_id(gold_spans_file):
    gold = load_gold(gold_spans_file)
    assert set(gold) == {"s1", "s2"}
    assert gold["s1"]["entities"][0]["type"] == "Condition"


def test_evaluate_spans_format_perfect_predictions(gold_spans_file, tmp_path):
    gold = load_gold(gold_spans_file)
    pred_path = tmp_path / "preds.jsonl"
    _write_jsonl(
        pred_path,
        [{"id": sid, "entities": rec["entities"]} for sid, rec in gold.items()],
    )
    result = evaluate(gold_spans_file, pred_path, pred_format="spans")
    assert result["exact"]["overall"]["f1"] == 1.0
    assert result["relaxed"]["overall"]["f1"] == 1.0
    assert result["coverage"]["num_missing_predictions"] == 0


def test_evaluate_reports_missing_predictions_without_crashing(gold_spans_file, tmp_path):
    pred_path = tmp_path / "preds.jsonl"
    _write_jsonl(pred_path, [{"id": "s1", "entities": []}])  # s2 missing entirely
    result = evaluate(gold_spans_file, pred_path, pred_format="spans")
    assert result["coverage"]["num_missing_predictions"] == 1
    assert result["coverage"]["missing_ids_sample"] == ["s2"]


def test_evaluate_bio_format_requires_tokens_path(gold_spans_file, tmp_path):
    pred_path = tmp_path / "preds.jsonl"
    _write_jsonl(pred_path, [])
    with pytest.raises(ValueError, match="--tokens is required"):
        evaluate(gold_spans_file, pred_path, pred_format="bio")


def test_evaluate_bio_format_raises_on_token_mismatch(gold_spans_file, tmp_path):
    tokens_path = tmp_path / "test.jsonl"
    _write_jsonl(
        tokens_path,
        [
            {
                "id": "s1",
                "tokens": ["type", "2", "diabetic", ",", "age", "18", "and", "over"],
                "ner_tags": ["O"] * 8,
            },
            {"id": "s2", "tokens": ["no", "known", "drug", "allergies"], "ner_tags": ["O"] * 4},
        ],
    )
    pred_path = tmp_path / "preds.jsonl"
    _write_jsonl(
        pred_path,
        [
            {"id": "s1", "tokens": ["totally", "different"], "predicted_tags": ["O", "O"]},
            {"id": "s2", "tokens": ["no", "known", "drug", "allergies"], "predicted_tags": ["O"] * 4},
        ],
    )
    with pytest.raises(ValueError, match="don't match gold tokens"):
        evaluate(gold_spans_file, pred_path, pred_format="bio", tokens_path=tokens_path)


def test_align_wordpiece_predictions_collapses_to_word_level():
    gold_tokens = ["diabetic", "mellitus"]
    model_tokens = ["diabet", "##ic", "mellitus"]
    model_tags = ["B-Condition", "I-Condition", "I-Condition"]
    word_tags = align_wordpiece_predictions(gold_tokens, model_tokens, model_tags)
    assert word_tags == ["B-Condition", "I-Condition"]


def test_align_wordpiece_predictions_raises_on_unalignable_input():
    with pytest.raises(ValueError):
        align_wordpiece_predictions(["diabetic"], ["totally", "different"], ["O", "O"])


def test_load_predictions_spans_strips_extra_fields(tmp_path):
    path = tmp_path / "preds.jsonl"
    _write_jsonl(
        path,
        [{"id": "s1", "entities": [{"type": "Drug", "start": 0, "end": 5, "confidence": 0.9}]}],
    )
    preds = load_predictions_spans(path)
    assert preds["s1"] == [{"type": "Drug", "start": 0, "end": 5}]
