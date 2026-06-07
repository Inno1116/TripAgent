"""Optional LangMem integration helpers."""

from __future__ import annotations

from importlib import import_module
from inspect import signature
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from kyuriagents.memory.types import MemoryScope

DEFAULT_LANGMEM_INSTRUCTIONS = """Store only durable, useful memories.
Prefer concise user preferences, facts, project rules, decisions, corrections,
and workflows. Never store secrets, passwords, API keys, or one-time transient
task details."""


def build_langmem_namespace(
    scope: MemoryScope,
    *,
    collection: str = "memories",
) -> tuple[str, ...]:
    """Build a tenant-isolated LangMem namespace.

    Args:
        scope: Memory operation scope.
        collection: Final namespace segment for the memory collection.

    Returns:
        Namespace tuple suitable for LangMem tools and LangGraph stores.
    """
    if scope.user_id is not None:
        return (scope.tenant_id, "users", scope.user_id, collection)
    return (scope.tenant_id, "tenant", collection)


def create_langmem_memory_tools(
    *,
    scope: MemoryScope | None = None,
    namespace: tuple[str, ...] | None = None,
    collection: str = "memories",
    instructions: str | None = None,
    store: object | None = None,
) -> list[object]:
    """Create LangMem manage/search tools without requiring LangMem at import time.

    Args:
        scope: Memory operation scope used to derive a namespace.
        namespace: Explicit LangMem namespace. Takes precedence over `scope`.
        collection: Namespace collection segment when deriving from `scope`.
        instructions: Optional extraction policy for the manage tool.
        store: Optional LangGraph store instance for LangMem.

    Returns:
        LangMem manage and search tools.

    Raises:
        ImportError: If `langmem` is not installed.
        ValueError: If neither `namespace` nor `scope` is provided.
    """
    resolved_namespace = namespace
    if resolved_namespace is None:
        if scope is None:
            msg = "Either `namespace` or `scope` must be provided."
            raise ValueError(msg)
        resolved_namespace = build_langmem_namespace(scope, collection=collection)

    module = _load_langmem()
    manage_factory = _resolve_factory(module, "create_manage_memory_tool")
    search_factory = _resolve_factory(module, "create_search_memory_tool")
    policy = instructions or DEFAULT_LANGMEM_INSTRUCTIONS

    return [
        manage_factory(**_factory_kwargs(manage_factory, namespace=resolved_namespace, instructions=policy, store=store)),
        search_factory(**_factory_kwargs(search_factory, namespace=resolved_namespace, instructions=None, store=store)),
    ]


def _load_langmem() -> ModuleType:
    try:
        return import_module("langmem")
    except ImportError as exc:
        msg = "Install `langmem` to create LangMem memory tools."
        raise ImportError(msg) from exc


def _resolve_factory(module: ModuleType, name: str) -> Callable[..., object]:
    factory = getattr(module, name, None)
    if factory is None:
        msg = f"`langmem.{name}` is not available."
        raise ImportError(msg)
    if not callable(factory):
        msg = f"`langmem.{name}` is not available."
        raise TypeError(msg)
    return cast("Callable[..., object]", factory)


def _factory_kwargs(
    factory: Callable[..., object],
    *,
    namespace: tuple[str, ...],
    instructions: str | None,
    store: object | None,
) -> dict[str, object]:
    parameters = signature(factory).parameters
    kwargs: dict[str, object] = {}
    if "namespace" in parameters:
        kwargs["namespace"] = namespace
    if "instructions" in parameters and instructions is not None:
        kwargs["instructions"] = instructions
    if "store" in parameters and store is not None:
        kwargs["store"] = store
    return kwargs
