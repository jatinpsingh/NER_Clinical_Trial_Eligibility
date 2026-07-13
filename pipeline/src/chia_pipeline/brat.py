"""Parser for CHIA's BRAT standoff (.txt/.ann) annotation format."""

import re
from dataclasses import dataclass

from .constants import KEPT_ENTITY_TYPES

_T_LINE_RE = re.compile(r"^(T\d+)\t(\S+) (.+?)\t(.*)$")


@dataclass(frozen=True)
class Fragment:
    """One contiguous character range belonging to an entity mention.

    CHIA represents discontinuous mentions (e.g. "major impairment of
    renal ... function") as a single T-line with multiple ';'-separated
    offset pairs. We split those into independent same-type fragments so
    each can be tagged as its own BIO span downstream.
    """

    entity_id: str
    type: str
    start: int
    end: int


def parse_ann(ann_text: str) -> list[Fragment]:
    """Parse a .ann file's contents into entity fragments.

    Only T-lines (text-bound annotations) are used; relation (R), attribute
    (A), equivalence (*), and note (#) lines are ignored since we only need
    entity spans for NER. Entities whose type isn't in KEPT_ENTITY_TYPES
    (the BRAT "ERROR" quality-flag category) are dropped.
    """
    fragments = []
    for line in ann_text.splitlines():
        if not line or not line.startswith("T"):
            continue
        m = _T_LINE_RE.match(line)
        if not m:
            continue
        entity_id, entity_type, offset_str, _text = m.groups()
        if entity_type not in KEPT_ENTITY_TYPES:
            continue
        for span in offset_str.split(";"):
            start_str, end_str = span.split()
            fragments.append(Fragment(entity_id, entity_type, int(start_str), int(end_str)))
    return fragments
