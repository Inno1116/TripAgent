"""DashScope model and embedding factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Protocol

    from langchain_core.language_models import BaseChatModel

    from kyuriagents.runtime.config import AgentRuntimeConfig

    class _ChatOpenAIConstructor(Protocol):
        def __call__(
            self,
            *,
            model: str,
            api_key: str,
            base_url: str,
            extra_body: Mapping[str, object] | None = None,
        ) -> BaseChatModel:
            """Create a chat model."""
            ...

    class _EmbeddingClient(Protocol):
        def embed_query(self, text: str) -> list[float]:
            """Embed a query string."""
            ...

    class _OpenAIEmbeddingsConstructor(Protocol):
        def __call__(
            self,
            *,
            model: str,
            dimensions: int | None,
            api_key: str,
            base_url: str,
            tiktoken_enabled: bool,
            check_embedding_ctx_length: bool,
        ) -> _EmbeddingClient:
            """Create an embedding client."""
            ...


EmbedQuery = Callable[[str], tuple[float, ...]]
"""Callable used by vector stores to embed a query."""


def create_dashscope_model(config: AgentRuntimeConfig, *, model_name: str | None = None) -> BaseChatModel:
    """Create a DashScope chat model through the OpenAI-compatible API.

    Args:
        config: Runtime configuration.
        model_name: Optional model override. When omitted, `config.chat_model`
            is used.

    Returns:
        LangChain chat model.

    Raises:
        ValueError: If `DASHSCOPE_API_KEY` is missing.
        ImportError: If `langchain-openai` is not installed.
    """
    if not config.dashscope_api_key:
        msg = "Set `DASHSCOPE_API_KEY` before creating a DashScope model."
        raise ValueError(msg)
    try:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `langchain-openai` or `kyuriagents[runtime]` to use DashScope models."
        raise ImportError(msg) from exc
    model_cls = cast("_ChatOpenAIConstructor", ChatOpenAI)
    extra_body = None
    if config.dashscope_enable_thinking is not None:
        extra_body = {"enable_thinking": config.dashscope_enable_thinking}
    return model_cls(
        model=model_name or config.chat_model,
        api_key=config.dashscope_api_key,
        base_url=config.dashscope_base_url,
        extra_body=extra_body,
    )


def create_dashscope_embed_query(config: AgentRuntimeConfig) -> EmbedQuery:
    """Create a DashScope query embedding function.

    Args:
        config: Runtime configuration.

    Returns:
        Callable returning a tuple embedding for one query.

    Raises:
        ValueError: If `DASHSCOPE_API_KEY` is missing.
        ImportError: If `langchain-openai` is not installed.
    """
    if not config.dashscope_api_key:
        msg = "Set `DASHSCOPE_API_KEY` before creating DashScope embeddings."
        raise ValueError(msg)
    try:
        from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `langchain-openai` or `kyuriagents[runtime]` to use DashScope embeddings."
        raise ImportError(msg) from exc

    embedding_cls = cast("_OpenAIEmbeddingsConstructor", OpenAIEmbeddings)
    embeddings = embedding_cls(
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
        api_key=config.dashscope_api_key,
        base_url=config.dashscope_base_url,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )

    def embed_query(query: str) -> tuple[float, ...]:
        return tuple(float(value) for value in embeddings.embed_query(query))

    return embed_query
