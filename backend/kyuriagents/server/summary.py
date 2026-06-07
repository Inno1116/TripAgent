"""Persistent rolling summary helpers for chat threads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from kyuriagents.runtime.dashscope import create_dashscope_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.language_models import BaseChatModel

    from kyuriagents.runtime import AgentRuntimeConfig
    from kyuriagents.server.identity import MessageRecord


class ThreadSummaryServiceProtocol(Protocol):
    """Contract for generating compact thread summaries."""

    def summarize(self, *, existing_summary: str, messages: Sequence[MessageRecord]) -> str:
        """Summarize compactable thread history."""
        ...


class ThreadSummaryService:
    """Generate persistent rolling summaries using the configured chat model."""

    def __init__(self, *, config: AgentRuntimeConfig, model: BaseChatModel | None = None) -> None:
        """Initialize the summary service.

        Args:
            config: Runtime configuration.
            model: Optional prebuilt summary model.
        """
        self._config = config
        self._model = model

    def summarize(self, *, existing_summary: str, messages: Sequence[MessageRecord]) -> str:
        """Summarize compactable thread history.

        Args:
            existing_summary: Previously persisted thread summary.
            messages: Historical messages that should be folded into the new summary.

        Returns:
            Updated summary text.
        """
        if not messages:
            return existing_summary
        model = self._model or create_dashscope_model(
            self._config,
            model_name=self._config.context_summary_model or self._config.chat_model,
        )
        result = model.invoke(
            [
                SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=_summary_user_prompt(existing_summary=existing_summary, messages=messages)),
            ]
        )
        content = getattr(result, "content", "")
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()


_SUMMARY_SYSTEM_PROMPT = """You maintain a compact persistent summary for an AI assistant conversation.

Update the summary with durable context only:
- user identity and preferences
- important project facts and decisions
- unresolved tasks, constraints, and assumptions
- tool or retrieval results only as conclusions with source hints

Do not preserve greetings, duplicated assistant phrasing, transient status text, or raw long tool output.
Return only the updated summary text.
"""


def _summary_user_prompt(*, existing_summary: str, messages: Sequence[MessageRecord]) -> str:
    existing = existing_summary.strip() or "(none)"
    return "\n\n".join(
        [
            "Existing summary:",
            existing,
            "New messages to fold into the summary:",
            _format_messages(messages),
        ]
    )


def _format_messages(messages: Sequence[MessageRecord]) -> str:
    lines: list[str] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"- {message.role}: {content}")
    return "\n".join(lines)


__all__ = ["ThreadSummaryService", "ThreadSummaryServiceProtocol"]
