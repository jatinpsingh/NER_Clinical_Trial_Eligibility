"""Lightweight offset-preserving tokenizer.

We deliberately avoid a heavyweight NLP dependency here: the model-specific
subword tokenizer (used later for PubMedBERT fine-tuning) will re-tokenize
these word-level tokens anyway via `is_split_into_words=True`. This just
needs to produce stable word boundaries with character offsets so entity
spans can be aligned to them.
"""

import re
from dataclasses import dataclass

# Words (including internal hyphens/apostrophes, e.g. "biopsy-proven",
# "patient's"), numbers with decimals (e.g. "2.5"), and standalone
# punctuation/symbols each as their own token.
_TOKEN_RE = re.compile(r"\d+\.\d+|[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
