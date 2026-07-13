from chia_pipeline.align import bio_to_spans, fragments_to_bio, resolve_overlaps
from chia_pipeline.brat import Fragment
from chia_pipeline.tokenizer import tokenize


def test_resolve_overlaps_keeps_longest():
    # "renal or hepatic function" — a 26-char Condition-like span fully
    # containing a shorter one starting at the same point.
    long_frag = Fragment("T1", "Temporal", 0, 26)
    short_frag = Fragment("T2", "Condition", 0, 5)
    kept = resolve_overlaps([long_frag, short_frag])
    assert kept == [long_frag]


def test_resolve_overlaps_keeps_disjoint_spans():
    a = Fragment("T1", "Condition", 0, 5)
    b = Fragment("T2", "Drug", 10, 15)
    assert resolve_overlaps([a, b]) == [a, b]


def test_fragments_to_bio_and_roundtrip_to_spans():
    text = "no kidney disease allowed"
    tokens = tokenize(text)
    frags = [Fragment("T1", "Negation", 0, 2), Fragment("T2", "Condition", 3, 17)]
    tags = fragments_to_bio(tokens, frags)
    assert tags[0] == "B-Negation"
    assert tags[1] == "B-Condition"  # "kidney"
    assert tags[2] == "I-Condition"  # "disease"
    assert tags[3] == "O"  # "allowed"

    spans = bio_to_spans(tokens, tags)
    assert spans == [
        {"type": "Negation", "start": 0, "end": 2},
        {"type": "Condition", "start": 3, "end": 17},
    ]
    assert text[spans[1]["start"] : spans[1]["end"]] == "kidney disease"


def test_bio_to_spans_breaks_on_type_change_without_b_tag():
    tokens = tokenize("a b")
    tags = ["B-Drug", "I-Condition"]  # malformed but must not merge silently
    spans = bio_to_spans(tokens, tags)
    assert len(spans) == 1
    assert spans[0]["type"] == "Drug"
