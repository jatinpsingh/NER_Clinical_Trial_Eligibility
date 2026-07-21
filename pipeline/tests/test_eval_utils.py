from chia_pipeline.eval_utils import score_corpus, score_corpus_relaxed, score_corpus_both


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


def test_relaxed_credits_partial_boundary_overlap():
    """'type 2 diabetes mellitus' (gold) vs 'diabetes mellitus' (pred, missed
    'type 2'): exact scores this as wrong; relaxed credits it as correct."""
    gold = [[{"type": "Condition", "start": 10, "end": 34}]]
    pred = [[{"type": "Condition", "start": 15, "end": 34}]]
    assert score_corpus(gold, pred)["overall"]["f1"] == 0.0
    assert score_corpus_relaxed(gold, pred)["overall"]["f1"] == 1.0


def test_relaxed_never_matches_wrong_type_even_with_full_overlap():
    gold = [[{"type": "Condition", "start": 0, "end": 5}]]
    pred = [[{"type": "Drug", "start": 0, "end": 5}]]
    assert score_corpus_relaxed(gold, pred)["overall"]["f1"] == 0.0


def test_relaxed_matching_is_one_to_one_not_many_to_one():
    """One long predicted span shouldn't get credit for both of two separate
    gold entities it happens to overlap."""
    gold = [[{"type": "Condition", "start": 0, "end": 5}, {"type": "Condition", "start": 10, "end": 15}]]
    pred = [[{"type": "Condition", "start": 0, "end": 15}]]
    result = score_corpus_relaxed(gold, pred)
    assert result["overall"]["support"] == 2
    assert result["per_type"]["Condition"]["recall"] == 0.5
    assert result["per_type"]["Condition"]["precision"] == 1.0


def test_score_corpus_both_returns_exact_and_relaxed():
    gold = [[{"type": "Condition", "start": 10, "end": 34}]]
    pred = [[{"type": "Condition", "start": 15, "end": 34}]]
    both = score_corpus_both(gold, pred)
    assert both["exact"]["overall"]["f1"] == 0.0
    assert both["relaxed"]["overall"]["f1"] == 1.0
