from chia_pipeline.eval_utils import score_corpus


def test_perfect_prediction_scores_one():
    gold = [[{"type": "Condition", "start": 0, "end": 5}]]
    result = score_corpus(gold, gold)
    assert result["overall"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}


def test_missed_and_spurious_entities():
    gold = [[{"type": "Condition", "start": 0, "end": 5}, {"type": "Drug", "start": 10, "end": 14}]]
    pred = [[{"type": "Condition", "start": 0, "end": 5}, {"type": "Drug", "start": 20, "end": 24}]]
    result = score_corpus(gold, pred)
    assert result["overall"]["precision"] == 0.5
    assert result["overall"]["recall"] == 0.5
    assert result["per_type"]["Condition"]["f1"] == 1.0
    assert result["per_type"]["Drug"]["f1"] == 0.0


def test_empty_gold_and_pred_have_zero_scores_not_errors():
    result = score_corpus([[]], [[]])
    assert result["overall"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
