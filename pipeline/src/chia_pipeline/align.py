"""Resolve overlapping entity fragments and align them to token-level BIO tags."""

from .brat import Fragment
from .tokenizer import Token


def resolve_overlaps(fragments: list[Fragment]) -> list[Fragment]:
    """Greedily keep the longest fragments, dropping any that overlap one
    already kept. BIO tagging is a flat scheme and CHIA's annotations are
    not (e.g. a longer Temporal or Condition span occasionally contains a
    shorter one), so a small number of entities are necessarily sacrificed
    here. (`Scope`, the dataset's dominant source of nesting, is excluded
    upstream entirely — see constants.py — rather than handled here.)
    """
    ordered = sorted(fragments, key=lambda f: (f.start - f.end, f.start))  # longest first
    kept: list[Fragment] = []
    for frag in ordered:
        if not any(frag.start < k.end and k.start < frag.end for k in kept):
            kept.append(frag)
    return sorted(kept, key=lambda f: f.start)


def fragments_to_bio(tokens: list[Token], fragments: list[Fragment]) -> list[str]:
    """Assign a BIO tag to each token from a set of non-overlapping fragments.

    A token is considered part of a fragment if their character ranges
    overlap at all — annotator span boundaries don't always land exactly on
    our tokenizer's word boundaries (e.g. a span ending mid-word), so exact
    equality would silently drop coverage.
    """
    tags = ["O"] * len(tokens)
    for frag in fragments:
        first = True
        for i, tok in enumerate(tokens):
            if tok.start < frag.end and frag.start < tok.end:
                tags[i] = f"{'B' if first else 'I'}-{frag.type}"
                first = False
    return tags


def bio_to_spans(tokens: list[Token], tags: list[str]) -> list[dict]:
    """Decode BIO tags back into character-offset entity spans (for the
    LLM-facing gold data and for shared entity-level evaluation), so both
    the PubMedBERT and GPT-4 pipelines are scored against the exact same
    flattened gold standard.
    """
    spans = []
    current = None
    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current:
                spans.append(current)
            current = {"type": tag[2:], "start": tok.start, "end": tok.end}
        elif tag.startswith("I-") and current and tag[2:] == current["type"]:
            current["end"] = tok.end
        else:
            if current:
                spans.append(current)
            current = None
    if current:
        spans.append(current)
    return spans
