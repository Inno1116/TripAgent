"""Token-aware context budgeting for Qwen-backed runtime calls."""

from __future__ import annotations

from functools import lru_cache
from math import ceil
from typing import TYPE_CHECKING, Any, Protocol, cast

from langchain_core.messages import BaseMessage, SystemMessage

if TYPE_CHECKING:
    from kyuriagents.runtime.config import AgentRuntimeConfig

_MESSAGE_OVERHEAD_TOKENS = 12
_TRUNCATION_SUFFIX = "\n\n[...truncated by context budget...]"
_CJK_START = 0x4E00
_CJK_END = 0x9FFF
_ASCII_LIMIT = 128


class TokenBudgetExceededError(ValueError):
    """Raised when a required input cannot fit the configured context window."""


TokenBudgetExceeded = TokenBudgetExceededError


class TokenCounter(Protocol):
    """Count tokens for runtime budgeting."""

    def count_text(self, text: str) -> int:
        """Count tokens in one text segment."""
        ...


class QwenTokenCounter:
    """Qwen tokenizer wrapper with conservative fallbacks."""

    def __init__(self, *, model_name: str, local_files_only: bool, strict: bool) -> None:
        """Initialize the token counter.

        Args:
            model_name: Hugging Face tokenizer model id or local path.
            local_files_only: Whether tokenizer loading may only use cached files.
            strict: Whether tokenizer loading failures should raise instead of
                using approximate fallbacks.
        """
        self._model_name = model_name
        self._local_files_only = local_files_only
        self._strict = strict
        self._tokenizer: object | None = None
        self._tiktoken_encoding: object | None = None
        self._load()

    def count_text(self, text: str) -> int:
        """Count text tokens."""
        if not text:
            return 0
        if self._tokenizer is not None:
            tokenizer = cast("Any", self._tokenizer)
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return len(tokenizer.encode(text))
        if self._tiktoken_encoding is not None:
            encoding = cast("Any", self._tiktoken_encoding)
            return len(encoding.encode(text))
        return _heuristic_tokens(text)

    def _load(self) -> None:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True,
                local_files_only=self._local_files_only,
            )
        except Exception as exc:
            if self._strict:
                msg = (
                    "Qwen tokenizer is required but could not be loaded. Install `transformers` "
                    f"and cache `{self._model_name}`, or disable strict tokenizer mode."
                )
                raise RuntimeError(msg) from exc
        else:
            self._tokenizer = tokenizer
            return

        try:
            import tiktoken  # noqa: PLC0415

            try:
                self._tiktoken_encoding = tiktoken.get_encoding("o200k_base")
            except Exception:  # noqa: BLE001 - older tiktoken releases may not include o200k_base
                self._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - final fallback is heuristic
            self._tiktoken_encoding = None


@lru_cache(maxsize=16)
def _cached_counter(model_name: str, local_files_only: int, strict: int) -> QwenTokenCounter:
    return QwenTokenCounter(model_name=model_name, local_files_only=bool(local_files_only), strict=bool(strict))


def token_counter_from_config(config: AgentRuntimeConfig) -> TokenCounter:
    """Build or reuse the configured Qwen token counter.

    Args:
        config: Runtime configuration.

    Returns:
        Reusable token counter.
    """
    return _cached_counter(config.tokenizer_model, int(config.tokenizer_local_files_only), int(config.tokenizer_strict))


def message_tokens(message: BaseMessage, counter: TokenCounter) -> int:
    """Estimate one chat message including role overhead.

    Args:
        message: LangChain message to estimate.
        counter: Token counter implementation.

    Returns:
        Estimated token count.
    """
    return counter.count_text(str(getattr(message, "type", "message"))) + counter.count_text(_message_text(message)) + _MESSAGE_OVERHEAD_TOKENS


def messages_tokens(messages: list[BaseMessage], counter: TokenCounter) -> int:
    """Estimate total chat message tokens.

    Args:
        messages: LangChain messages.
        counter: Token counter implementation.

    Returns:
        Estimated token count.
    """
    return sum(message_tokens(message, counter) for message in messages)


def enforce_user_input_budget(config: AgentRuntimeConfig, text: str, counter: TokenCounter) -> None:
    """Reject one user input that cannot be safely sent as a chat turn.

    Args:
        config: Runtime configuration.
        text: User-provided message.
        counter: Token counter implementation.

    Raises:
        TokenBudgetExceeded: If the message exceeds `max_user_input_tokens`.
    """
    tokens = counter.count_text(text)
    if tokens <= config.max_user_input_tokens:
        return
    msg = (
        f"Your input is too large for one chat turn ({tokens} tokens, "
        f"limit {config.max_user_input_tokens}). Please upload it as a knowledge-base document."
    )
    raise TokenBudgetExceeded(msg)


def fit_messages_to_context_budget(
    config: AgentRuntimeConfig,
    messages: list[BaseMessage],
    counter: TokenCounter,
) -> list[BaseMessage]:
    """Fit messages into the configured input budget by dropping oldest history.

    Args:
        config: Runtime configuration.
        messages: Messages to fit.
        counter: Token counter implementation.

    Returns:
        Messages that fit the configured budget.

    Raises:
        TokenBudgetExceeded: If required current messages exceed the budget.
    """
    if not messages:
        return []
    budget = _input_token_budget(config)
    if messages_tokens(messages, counter) <= budget:
        return messages

    system = messages[0] if isinstance(messages[0], SystemMessage) else None
    rest = messages[1:] if system is not None else messages
    latest = rest[-1:] if rest else []
    middle = rest[:-1] if latest else rest
    kept: list[BaseMessage] = [*latest]
    used = messages_tokens(([system] if system is not None else []) + kept, counter)
    for message in reversed(middle):
        cost = message_tokens(message, counter)
        if used + cost > budget:
            continue
        kept.insert(0, message)
        used += cost
    fitted = ([system] if system is not None else []) + kept
    if messages_tokens(fitted, counter) > budget:
        msg = "The required system prompt and current user input exceed the configured context budget."
        raise TokenBudgetExceeded(msg)
    return fitted


def trim_text_to_token_budget(text: str, *, counter: TokenCounter, max_tokens: int) -> str:
    """Trim text to a token budget, preserving a suffix marker.

    Args:
        text: Text to trim.
        counter: Token counter implementation.
        max_tokens: Maximum output tokens.

    Returns:
        Original or trimmed text.
    """
    if max_tokens <= 0 or counter.count_text(text) <= max_tokens:
        return text
    suffix_tokens = counter.count_text(_TRUNCATION_SUFFIX)
    budget = max(max_tokens - suffix_tokens, 1)
    lo = 0
    hi = len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        if counter.count_text(candidate) <= budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best.rstrip() + _TRUNCATION_SUFFIX


def _input_token_budget(config: AgentRuntimeConfig) -> int:
    hard_window = int(config.context_window_tokens * config.context_safety_ratio)
    return max(hard_window - config.reserved_output_tokens, 1)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _heuristic_tokens(text: str) -> int:
    cjk = 0
    ascii_chars = 0
    other = 0
    for char in text:
        codepoint = ord(char)
        if _CJK_START <= codepoint <= _CJK_END:
            cjk += 1
        elif codepoint < _ASCII_LIMIT:
            ascii_chars += 1
        else:
            other += 1
    return ceil(cjk * 1.2 + ascii_chars / 4 + other / 2)


__all__ = [
    "QwenTokenCounter",
    "TokenBudgetExceeded",
    "TokenBudgetExceededError",
    "TokenCounter",
    "enforce_user_input_budget",
    "fit_messages_to_context_budget",
    "message_tokens",
    "messages_tokens",
    "token_counter_from_config",
    "trim_text_to_token_budget",
]
