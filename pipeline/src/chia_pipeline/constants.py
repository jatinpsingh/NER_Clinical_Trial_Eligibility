"""Entity label schema for the CHIA NER pipeline.

CHIA's annotation.conf groups entity types into three buckets: CONCEPTS
(the actual clinical entities), ANNOTATION (modifiers like negation/value),
and ERROR (annotator data-quality flags such as "Non-representable" or
"Parsing_Error" that are not real entities). We only keep CONCEPTS +
ANNOTATION for NER, matching prior CHIA NER work (e.g. Gao et al. 2022).

We source from the "chia_without_scope" release (see download.py), not
"chia_with_scope": `Scope` is a coordination/relation-boundary annotation
layered on top of the original 15-type schema, not a standalone clinical
entity, and it is almost always the longest span in any overlap group. Under
flat BIO + keep-longest overlap resolution it systematically evicted the
entities nested inside it — in the with-scope release, ~14.5k of ~15.9k
dropped entity mentions were dropped specifically because a Scope span won
the overlap, including ~5.2k Condition and ~2.4k Drug mentions. Dropping
Scope entirely (matching the schema prior CHIA NER baselines used) raised
overall entity retention from ~63% to ~91.5%.
"""

CONCEPT_TYPES = [
    "Person",
    "Condition",
    "Drug",
    "Observation",
    "Measurement",
    "Procedure",
    "Device",
    "Visit",
]

ANNOTATION_TYPES = [
    "Negation",
    "Qualifier",
    "Temporal",
    "Value",
    "Multiplier",
    "Reference_point",
    "Line",
    "Mood",
]

# Allowlist, not a blocklist: the raw data also contains error/quality-flag
# tags (Non-query-able, Parsing_Error, Undefined_semantics, ...) that aren't
# in annotation.conf's documented ERROR list either, so we only keep types
# we explicitly recognize rather than trying to enumerate every reject.
KEPT_ENTITY_TYPES = CONCEPT_TYPES + ANNOTATION_TYPES

LABELS = ["O"] + [f"{p}-{t}" for t in KEPT_ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
SPLIT_SEED = 42
