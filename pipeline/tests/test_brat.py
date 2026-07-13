from chia_pipeline.brat import parse_ann


def test_parses_simple_entity():
    ann = "T1\tCondition 28 55\tmetastatic carcinoid tumors\n"
    frags = parse_ann(ann)
    assert len(frags) == 1
    assert frags[0].type == "Condition"
    assert (frags[0].start, frags[0].end) == (28, 55)


def test_splits_discontinuous_span_into_fragments():
    ann = "T19\tCondition 331 356;368 376\tmajor impairment of renal function\n"
    frags = parse_ann(ann)
    assert [(f.start, f.end) for f in frags] == [(331, 356), (368, 376)]
    assert all(f.type == "Condition" and f.entity_id == "T19" for f in frags)


def test_drops_error_category_entities():
    ann = "T1\tNon-representable 0 5\tabcde\n"
    assert parse_ann(ann) == []


def test_ignores_relation_and_attribute_lines():
    ann = (
        "T1\tCondition 0 5\tabcde\n"
        "R1\tHas_value Arg1:T1 Arg2:T2\t\n"
        "A1\tOptional T1\n"
        "*\tOR T1 T2\n"
    )
    frags = parse_ann(ann)
    assert len(frags) == 1
