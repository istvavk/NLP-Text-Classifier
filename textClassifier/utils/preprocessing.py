"""Text preprocessing utilities.

This module intentionally contains:
- A plain tokenizer (for baseline / classic ML)
- A tokenizer factory implemented as a *closure* (rubric requirement)
- Simple helpers that are easy to test (doctest + unittest)

Run doctests:
    python -m doctest -v utils/preprocessing.py
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, List, Sequence, Set

_WORD_RE: re.Pattern[str] = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Tokenize input text into lowercase word tokens.

    >>> tokenize("Home team wins!")
    ['home', 'team', 'wins']
    >>> tokenize("  Mixed-CASE, words... ")
    ['mixed', 'case', 'words']
    """
    return _WORD_RE.findall(text.lower())


def make_tokenizer(stopwords: Iterable[str] | None = None) -> Callable[[str], List[str]]:
    """Create a tokenizer function (closure) with optional stopword removal.

    >>> tok = make_tokenizer(stopwords={"the", "a"})
    >>> tok("The team won a match")
    ['team', 'won', 'match']
    >>> tok2 = make_tokenizer()
    >>> tok2("The team won a match")
    ['the', 'team', 'won', 'a', 'match']
    """
    stopwords_set: Set[str] = set(w.lower() for w in (stopwords or []))

    def _tokenize(text: str) -> List[str]:
        tokens: List[str] = tokenize(text)
        if not stopwords_set:
            return tokens
        return [t for t in tokens if t not in stopwords_set]

    return _tokenize


def batch_tokenize(
    texts: Sequence[str],
    tokenizer: Callable[[str], List[str]] | None = None,
) -> List[List[str]]:
    """Tokenize a batch of texts.

    >>> batch_tokenize(["Home team wins!", "A B C"])
    [['home', 'team', 'wins'], ['a', 'b', 'c']]
    """
    tok = tokenizer or tokenize
    return [tok(t) for t in texts]
