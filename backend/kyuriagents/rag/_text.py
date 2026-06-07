"""Small text helpers for deterministic retrieval tests and fallbacks."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize text for lightweight lexical scoring."""
    return tuple(match.group(0).lower() for match in _TOKEN_RE.finditer(text))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two vectors."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
