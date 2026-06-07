"""Budget-aware memory context compression."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kyuriagents.memory.types import MemoryRecord, MemorySearchResult

_DEFAULT_MAX_MEMORIES = 5
_DEFAULT_MAX_CONTEXT_CHARS = 2_400
_DEFAULT_MAX_MEMORY_CHARS = 480
_ELLIPSIS = "...[truncated]"


@dataclass(frozen=True, kw_only=True)
class MemoryContextBudget:
    """Budget controls for memory context injection.

    Args:
        max_memories: Maximum memory records to include.
        max_context_chars: Approximate total character budget.
        max_memory_chars: Maximum characters per memory text.
    """

    max_memories: int = _DEFAULT_MAX_MEMORIES
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS
    max_memory_chars: int = _DEFAULT_MAX_MEMORY_CHARS

    def __post_init__(self) -> None:
        """Validate positive budget values."""
        for name, value in (
            ("max_memories", self.max_memories),
            ("max_context_chars", self.max_context_chars),
            ("max_memory_chars", self.max_memory_chars),
        ):
            if value <= 0:
                msg = f"`{name}` must be positive."
                raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class CompressedMemoryContext:
    """Compressed memory results and accounting metadata.

    Args:
        results: Results selected for prompt injection.
        omitted_count: Number of relevant memories omitted due to budget.
        truncated_count: Number of selected memories whose text was truncated.
    """

    results: tuple[MemorySearchResult, ...]
    omitted_count: int = 0
    truncated_count: int = 0


class MemoryContextCompressor:
    """Select and trim memories before prompt injection."""

    def __init__(self, budget: MemoryContextBudget | None = None) -> None:
        """Initialize the compressor.

        Args:
            budget: Optional context budget.
        """
        self._budget = budget or MemoryContextBudget()

    @property
    def budget(self) -> MemoryContextBudget:
        """Return the active memory context budget."""
        return self._budget

    def compress(self, results: list[MemorySearchResult]) -> CompressedMemoryContext:
        """Compress ranked memory search results to fit the budget.

        Args:
            results: Ranked or partially ranked memory results.

        Returns:
            Selected and trimmed memory results.
        """
        selected: list[MemorySearchResult] = []
        seen: set[str] = set()
        used_chars = 0
        truncated_count = 0

        for result in sorted(results, key=_ranking_key, reverse=True):
            memory = result.memory
            if memory.memory_id in seen:
                continue
            seen.add(memory.memory_id)
            if len(selected) >= self._budget.max_memories:
                break

            text = memory.summary or memory.content
            remaining = self._budget.max_context_chars - used_chars
            if remaining <= 0:
                break
            allowed = min(self._budget.max_memory_chars, remaining)
            compact_text, truncated = _truncate(text, allowed)
            compact_memory = _with_compact_text(memory, compact_text)
            selected.append(replace(result, memory=compact_memory))
            used_chars += len(compact_text)
            if truncated:
                truncated_count += 1

        omitted_count = max(0, len({result.memory.memory_id for result in results}) - len(selected))
        return CompressedMemoryContext(
            results=tuple(selected),
            omitted_count=omitted_count,
            truncated_count=truncated_count,
        )


def _ranking_key(result: MemorySearchResult) -> tuple[float, float, float, str]:
    memory = result.memory
    return (
        result.score,
        memory.importance,
        memory.confidence,
        memory.updated_at or memory.created_at or memory.memory_id,
    )


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized, False
    if limit <= len(_ELLIPSIS):
        return _ELLIPSIS[:limit], True
    return normalized[: limit - len(_ELLIPSIS)].rstrip() + _ELLIPSIS, True


def _with_compact_text(memory: MemoryRecord, text: str) -> MemoryRecord:
    if memory.summary:
        return replace(memory, summary=text)
    return replace(memory, content=text)


__all__ = [
    "CompressedMemoryContext",
    "MemoryContextBudget",
    "MemoryContextCompressor",
]
