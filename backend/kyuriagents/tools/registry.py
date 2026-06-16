"""Registry for native, built-in, runtime, and MCP tool descriptors."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from langchain_core.tools import BaseTool

from kyuriagents.tools.types import ToolDescriptor, ToolRisk, ToolSource

_READ_ONLY_BUILTINS = (
    "ls",
    "read_file",
    "glob",
    "grep",
    "search_knowledge_base",
    "get_travel_profile",
    "estimate_travel_budget",
)
_EXTERNAL_READ_BUILTINS = (
    "web_search",
    "web_research",
    "web_fetch_page",
    "web_fetch_static",
    "web_render_page",
    "web_agent",
    "rag_agent",
    "amap_search_poi",
    "amap_get_weather",
    "amap_plan_route",
    "amap_get_poi_detail",
    "amap_create_trip_map",
)
_WRITE_BUILTINS = (
    "write_todos",
    "write_file",
    "edit_file",
    "update_travel_profile",
)
_DESTRUCTIVE_BUILTINS = ("execute",)
_RUNTIME_BUILTINS = ("task", "launch_task", "check_task", "update_task", "cancel_task", "list_tasks")


@dataclass
class ToolRegistry:
    """Mutable registry of tool descriptors.

    Args:
        descriptors: Optional descriptors to seed the registry.
        unknown_tool_risk: Risk assigned to tools without an explicit descriptor.
    """

    descriptors: dict[str, ToolDescriptor] = field(default_factory=dict)
    unknown_tool_risk: ToolRisk = "read_only"

    def register(self, descriptor: ToolDescriptor, *, replace_existing: bool = False) -> ToolDescriptor:
        """Register one descriptor.

        Args:
            descriptor: Descriptor to add.
            replace_existing: Whether an existing descriptor may be replaced.

        Returns:
            Registered descriptor.

        Raises:
            ValueError: If the name already exists and replacement is disabled.
        """
        if descriptor.name in self.descriptors and not replace_existing:
            msg = f"Tool descriptor `{descriptor.name}` is already registered."
            raise ValueError(msg)
        self.descriptors[descriptor.name] = descriptor
        return descriptor

    def register_many(self, descriptors: Iterable[ToolDescriptor], *, replace_existing: bool = False) -> None:
        """Register multiple descriptors.

        Args:
            descriptors: Descriptors to add.
            replace_existing: Whether existing descriptors may be replaced.
        """
        for descriptor in descriptors:
            self.register(descriptor, replace_existing=replace_existing)

    def register_tool(
        self,
        tool: BaseTool | Callable | Mapping[str, object],
        *,
        risk: ToolRisk = "read_only",
        source: ToolSource = "native",
        requires_confirmation: bool = False,
        replace_existing: bool = False,
    ) -> ToolDescriptor:
        """Create and register a descriptor from a concrete tool object.

        Args:
            tool: LangChain tool, callable, or tool dictionary.
            risk: Risk class for this tool.
            source: Tool origin.
            requires_confirmation: Whether to block until approval is wired.
            replace_existing: Whether existing descriptors may be replaced.

        Returns:
            Registered descriptor.
        """
        name = tool_name(tool)
        description = tool_description(tool)
        return self.register(
            ToolDescriptor(
                name=name,
                description=description,
                risk=risk,
                source=source,
                requires_confirmation=requires_confirmation,
            ),
            replace_existing=replace_existing,
        )

    def descriptor_for(self, tool_name: str) -> ToolDescriptor:
        """Return a descriptor for a runtime tool name.

        Args:
            tool_name: Tool name from a tool call.

        Returns:
            Explicit descriptor, built-in descriptor, or a conservative fallback.
        """
        descriptor = self.descriptors.get(tool_name)
        if descriptor is not None:
            return descriptor
        return ToolDescriptor(
            name=tool_name,
            description="Unregistered tool.",
            risk=self.unknown_tool_risk,
            source="native",
        )

    def copy(self) -> ToolRegistry:
        """Return a shallow copy of the registry."""
        return ToolRegistry(descriptors=dict(self.descriptors), unknown_tool_risk=self.unknown_tool_risk)


def default_tool_registry(*, unknown_tool_risk: ToolRisk = "read_only") -> ToolRegistry:
    """Create a registry with KyuriAgents built-in tools classified.

    Args:
        unknown_tool_risk: Fallback risk for user-supplied tools.

    Returns:
        Registry seeded with built-in descriptors.
    """
    registry = ToolRegistry(unknown_tool_risk=unknown_tool_risk)
    registry.register_many(_builtin_descriptors(), replace_existing=True)
    return registry


def merge_tool_sequences(
    *tool_groups: Sequence[BaseTool | Callable | dict[str, Any]] | None,
) -> list[BaseTool | Callable | dict[str, Any]]:
    """Merge optional tool sequences while preserving order.

    Args:
        tool_groups: Tool sequences to merge.

    Returns:
        Flat list of tools.
    """
    merged: list[BaseTool | Callable | dict[str, Any]] = []
    for group in tool_groups:
        if group:
            merged.extend(group)
    return merged


def tool_name(tool: BaseTool | Callable | Mapping[str, object]) -> str:
    """Resolve a tool name from a supported tool object.

    Args:
        tool: LangChain tool, callable, or tool dictionary.

    Returns:
        Runtime tool name.

    Raises:
        ValueError: If no name can be resolved.
    """
    if isinstance(tool, BaseTool):
        return tool.name
    if isinstance(tool, Mapping):
        mapping = cast("Mapping[str, object]", tool)
        raw = mapping.get("name")
        if isinstance(raw, str) and raw:
            return raw
    raw_name = getattr(tool, "__name__", "")
    if isinstance(raw_name, str) and raw_name:
        return raw_name
    msg = "Tool name could not be resolved."
    raise ValueError(msg)


def tool_description(tool: BaseTool | Callable | Mapping[str, object]) -> str:
    """Resolve a short tool description.

    Args:
        tool: LangChain tool, callable, or tool dictionary.

    Returns:
        Description when available, otherwise an empty string.
    """
    if isinstance(tool, BaseTool):
        return str(tool.description or "")
    if isinstance(tool, Mapping):
        mapping = cast("Mapping[str, object]", tool)
        raw = mapping.get("description")
        return raw if isinstance(raw, str) else ""
    doc = getattr(tool, "__doc__", None)
    return doc.strip() if isinstance(doc, str) and doc.strip() else ""


def apply_descriptor_overrides(
    descriptors: Iterable[ToolDescriptor],
    overrides: Mapping[str, ToolDescriptor],
) -> list[ToolDescriptor]:
    """Apply descriptor overrides by name.

    Args:
        descriptors: Base descriptors.
        overrides: Replacements keyed by name.

    Returns:
        Descriptors with matching overrides merged in.
    """
    resolved: list[ToolDescriptor] = []
    for descriptor in descriptors:
        override = overrides.get(descriptor.name)
        if override is None:
            resolved.append(descriptor)
            continue
        resolved.append(
            replace(
                descriptor,
                description=override.description or descriptor.description,
                risk=override.risk,
                source=override.source,
                requires_confirmation=override.requires_confirmation,
                timeout_seconds=override.timeout_seconds or descriptor.timeout_seconds,
                tags=override.tags or descriptor.tags,
                metadata={**descriptor.metadata, **override.metadata},
            )
        )
    return resolved


def _builtin_descriptors() -> list[ToolDescriptor]:
    return [
        *[
            ToolDescriptor(
                name=name,
                description="Read-only KyuriAgents built-in tool.",
                risk="read_only",
                source="builtin",
            )
            for name in _READ_ONLY_BUILTINS
        ],
        *[
            ToolDescriptor(
                name=name,
                description="External read KyuriAgents runtime tool.",
                risk="external_read",
                source="runtime",
            )
            for name in _EXTERNAL_READ_BUILTINS
        ],
        *[
            ToolDescriptor(
                name=name,
                description="Write-capable KyuriAgents built-in tool.",
                risk="write",
                source="builtin",
                requires_confirmation=name in {"write_file", "edit_file"},
            )
            for name in _WRITE_BUILTINS
        ],
        *[
            ToolDescriptor(
                name=name,
                description="High-risk KyuriAgents built-in tool.",
                risk="destructive",
                source="builtin",
                requires_confirmation=True,
            )
            for name in _DESTRUCTIVE_BUILTINS
        ],
        *[
            ToolDescriptor(
                name=name,
                description="Runtime orchestration tool.",
                risk="external_read",
                source="runtime",
            )
            for name in _RUNTIME_BUILTINS
        ],
    ]


__all__ = [
    "ToolRegistry",
    "apply_descriptor_overrides",
    "default_tool_registry",
    "merge_tool_sequences",
    "tool_description",
    "tool_name",
]
