"""Smoke test a real Agent RAG tool call against configured services."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from kyuriagents.runtime import AgentRuntimeConfig, create_kyuri_agent

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

_LOGGER = logging.getLogger("rag_agent_smoke")
_DEFAULT_KB_ID = "stratrag:train:train_000000"
_DEFAULT_QUERY = (
    "请分别调用 search_knowledge_base 查询 `Arthur's Magazine start date` 和 `First for Women start date`, 然后回答 Which magazine was started first?"
)
_PREVIEW_LIMIT = 500


def main() -> None:
    """Run a real Agent RAG smoke test."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Verify that the real agent can call the RAG tool.")
    parser.add_argument("--kb-id", default=_DEFAULT_KB_ID, help="Knowledge base id to scope RAG retrieval.")
    parser.add_argument("--query", default=_DEFAULT_QUERY, help="User query sent to the agent.")
    parser.add_argument("--thread-id", default="rag-agent-smoke", help="Thread id used for audit lookup.")
    args = parser.parse_args()

    config = _runtime_config(kb_id=args.kb_id, thread_id=args.thread_id)
    before = _audit_count(config, args.thread_id)
    agent = create_kyuri_agent(
        config,
        system_prompt=(
            "You are validating RAG tool use. You must call `search_knowledge_base` "
            "before answering any factual question. When the user asks for separate "
            "lookups, call `search_knowledge_base` once per lookup."
        ),
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=args.query)]},
        config={
            "recursion_limit": 20,
            "configurable": {
                "tenant_id": config.tenant_id,
                "user_id": config.user_id,
                "thread_id": args.thread_id,
                "tool_thread_id": args.thread_id,
                "rag_kb_ids": [args.kb_id],
            },
        },
    )
    messages = cast("list[BaseMessage]", result["messages"])
    tool_calls = _tool_calls(messages)
    tool_messages = [message for message in messages if message.type == "tool"]
    after = _audit_count(config, args.thread_id)

    _LOGGER.info("tool_call_count=%s", len(tool_calls))
    for index, tool_call in enumerate(tool_calls, start=1):
        _LOGGER.info("tool_call[%s]=%s args=%s", index, tool_call.get("name"), tool_call.get("args"))
    _LOGGER.info("tool_message_count=%s", len(tool_messages))
    for index, message in enumerate(tool_messages, start=1):
        _LOGGER.info("tool_message[%s]=%s", index, _preview(str(message.text)))
    _LOGGER.info("audit_delta=%s", None if before is None or after is None else after - before)
    _LOGGER.info("final=%s", _preview(str(messages[-1].text)))


def _runtime_config(*, kb_id: str, thread_id: str) -> AgentRuntimeConfig:
    config = AgentRuntimeConfig.from_env()
    return replace(
        config,
        rag_mode=cast("RetrievalMode", "tool"),
        enable_travel_profile=False,
        enable_checkpointer=False,
        rag_kb_ids=(kb_id,),
        thread_id=thread_id,
    )


def _tool_calls(messages: list[BaseMessage]) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for message in messages:
        raw = getattr(message, "tool_calls", None)
        if raw:
            calls.extend(cast("list[dict[str, object]]", raw))
    return calls


def _audit_count(config: AgentRuntimeConfig, thread_id: str) -> int | None:
    if not config.postgres_dsn:
        return None
    try:
        import psycopg  # noqa: PLC0415
    except ImportError:
        return None
    with psycopg.connect(config.postgres_dsn) as connection:
        row = connection.execute(
            "SELECT count(*) FROM agent_tool_calls WHERE thread_id = %s AND tool_name = 'search_knowledge_base'",
            (thread_id,),
        ).fetchone()
    if row is None:
        return None
    return int(row[0])


def _preview(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= _PREVIEW_LIMIT:
        return normalized
    return normalized[: _PREVIEW_LIMIT - 14] + "...[truncated]"


if __name__ == "__main__":
    main()
