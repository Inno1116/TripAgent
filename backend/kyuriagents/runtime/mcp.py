"""MCP loading helpers for runtime deployments."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

from kyuriagents.tools.registry import tool_description, tool_name
from kyuriagents.tools.types import ToolDescriptor, ToolRisk

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from langchain_core.tools import BaseTool

    from kyuriagents.runtime.config import AgentRuntimeConfig

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_GOVERNANCE_KEYS = frozenset(
    {
        "description",
        "display_name",
        "metadata",
        "requires_confirmation",
        "risk",
        "tags",
        "timeout_seconds",
        "tool_confirmation",
        "tool_risks",
    }
)


@dataclass(frozen=True, kw_only=True)
class MCPServerRuntimeConfig:
    """Runtime MCP server configuration with governance metadata.

    Args:
        name: Server identifier.
        connection: Connection dictionary passed to `MultiServerMCPClient`.
        risk: Default risk for tools from this server.
        requires_confirmation: Whether server tools require confirmation.
        tool_risks: Optional per-tool risk overrides.
        tool_confirmation: Optional per-tool confirmation overrides.
        tags: Classification tags.
        metadata: Additional JSON-serializable metadata.
    """

    name: str
    connection: dict[str, object]
    risk: ToolRisk = "external_read"
    requires_confirmation: bool = False
    tool_risks: dict[str, ToolRisk] = field(default_factory=dict)
    tool_confirmation: dict[str, bool] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class MCPRuntimeConfig:
    """Parsed MCP configuration.

    Args:
        servers: Server configs keyed by server name.
        tool_name_prefix: Whether descriptor matching expects server-prefixed tool names.
    """

    servers: dict[str, MCPServerRuntimeConfig]
    tool_name_prefix: bool = False


@dataclass(frozen=True, kw_only=True)
class LoadedMCPTools:
    """Loaded MCP tools and descriptors.

    Args:
        tools: LangChain tools returned by MCP adapters.
        descriptors: Governance descriptors for the tools.
        client: Underlying MCP client, retained for tool closures.
    """

    tools: list[BaseTool]
    descriptors: tuple[ToolDescriptor, ...]
    client: object


def load_mcp_config(path: str | Path, *, tool_name_prefix: bool = False, env: Mapping[str, str] | None = None) -> MCPRuntimeConfig:
    """Load MCP server config from a JSON file.

    Args:
        path: JSON config path.
        tool_name_prefix: Whether tools should be server-prefixed.
        env: Environment mapping used for `${VAR}` interpolation.

    Returns:
        Parsed MCP runtime config.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "MCP config must be a JSON object."
        raise TypeError(msg)
    source = env if env is not None else os.environ
    expanded = cast("dict[str, object]", _expand_env(raw, source))
    servers_raw = expanded.get("servers", expanded)
    if not isinstance(servers_raw, dict):
        msg = "MCP config `servers` must be an object."
        raise TypeError(msg)

    servers: dict[str, MCPServerRuntimeConfig] = {}
    for name, value in servers_raw.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            msg = "Each MCP server entry must be an object keyed by server name."
            raise TypeError(msg)
        servers[name] = _server_config(name, cast("dict[str, object]", value))

    return MCPRuntimeConfig(servers=servers, tool_name_prefix=tool_name_prefix)


async def aload_mcp_tools(config: AgentRuntimeConfig) -> LoadedMCPTools:
    """Load LangChain tools from configured MCP servers.

    Args:
        config: Runtime configuration.

    Returns:
        Loaded tools plus governance descriptors.

    Raises:
        ValueError: If MCP is enabled without `DEEPAGENTS_MCP_CONFIG_PATH`.
        ImportError: If `langchain-mcp-adapters` is not installed.
    """
    if not config.mcp_config_path:
        msg = "Set `DEEPAGENTS_MCP_CONFIG_PATH` before enabling MCP."
        raise ValueError(msg)
    try:
        client_module = import_module("langchain_mcp_adapters.client")
    except ImportError as exc:
        msg = "Install `langchain-mcp-adapters` or `kyuriagents[runtime]` to use MCP tools."
        raise ImportError(msg) from exc
    client_cls = client_module.MultiServerMCPClient

    mcp_config = load_mcp_config(config.mcp_config_path, tool_name_prefix=config.mcp_tool_name_prefix)
    client = client_cls(
        {name: server.connection for name, server in mcp_config.servers.items()},
    )
    tools = await client.get_tools()
    descriptors = build_mcp_tool_descriptors(tools, mcp_config)
    return LoadedMCPTools(tools=list(tools), descriptors=descriptors, client=client)


def load_mcp_tools(config: AgentRuntimeConfig) -> LoadedMCPTools:
    """Synchronously load MCP tools during startup.

    Args:
        config: Runtime configuration.

    Returns:
        Loaded MCP tools.

    Raises:
        RuntimeError: If called while an event loop is already running.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aload_mcp_tools(config))
    msg = "Use `await aload_mcp_tools(config)` when startup already runs inside an event loop."
    raise RuntimeError(msg)


def build_mcp_tool_descriptors(tools: list[BaseTool], config: MCPRuntimeConfig) -> tuple[ToolDescriptor, ...]:
    """Build governance descriptors for loaded MCP tools.

    Args:
        tools: Loaded MCP tools.
        config: Parsed MCP runtime config.

    Returns:
        Tool descriptors.
    """
    descriptors: list[ToolDescriptor] = []
    for tool in tools:
        name = tool_name(tool)
        server_name, raw_name = _split_server_tool_name(name, config.servers.keys(), prefixed=config.tool_name_prefix)
        server = config.servers.get(server_name) if server_name else None
        if server is None:
            descriptors.append(
                ToolDescriptor(
                    name=name,
                    description=tool_description(tool),
                    risk="external_read",
                    source="mcp",
                    metadata={"mcp_server": server_name or ""},
                )
            )
            continue
        risk = server.tool_risks.get(name) or server.tool_risks.get(raw_name) or server.risk
        requires_confirmation = server.tool_confirmation.get(name, server.tool_confirmation.get(raw_name, server.requires_confirmation))
        descriptors.append(
            ToolDescriptor(
                name=name,
                description=tool_description(tool),
                risk=risk,
                source="mcp",
                requires_confirmation=requires_confirmation,
                timeout_seconds=_optional_int(server.connection.get("timeout_seconds")),
                tags=server.tags,
                metadata={"mcp_server": server.name, **server.metadata},
            )
        )
    return tuple(descriptors)


def _server_config(name: str, value: dict[str, object]) -> MCPServerRuntimeConfig:
    connection = {key: item for key, item in value.items() if key not in _GOVERNANCE_KEYS}
    if "transport" not in connection:
        msg = f"MCP server `{name}` must define `transport`."
        raise ValueError(msg)
    return MCPServerRuntimeConfig(
        name=name,
        connection=connection,
        risk=_risk(value.get("risk"), default="external_read"),
        requires_confirmation=_bool(value.get("requires_confirmation"), default=False),
        tool_risks=_tool_risks(value.get("tool_risks")),
        tool_confirmation=_tool_confirmation(value.get("tool_confirmation")),
        tags=_string_tuple(value.get("tags")),
        metadata=_object_dict(value.get("metadata")),
    )


def _expand_env(value: object, env: Mapping[str, str]) -> object:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: env.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(item, env) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item, env) for key, item in value.items()}
    return value


def _risk(value: object, *, default: ToolRisk) -> ToolRisk:
    if value is None or value == "":
        return default
    if value not in _TOOL_RISKS:
        allowed = ", ".join(_TOOL_RISKS)
        msg = f"`ToolRisk` must be one of: {allowed}."
        raise ValueError(msg)
    return cast("ToolRisk", value)


def _tool_risks(value: object) -> dict[str, ToolRisk]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = "`tool_risks` must be an object."
        raise TypeError(msg)
    return {str(key): _risk(item, default="external_read") for key, item in value.items()}


def _tool_confirmation(value: object) -> dict[str, bool]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = "`tool_confirmation` must be an object."
        raise TypeError(msg)
    return {str(key): _bool(item, default=False) for key, item in value.items()}


def _split_server_tool_name(tool_name: str, server_names: Iterable[str], *, prefixed: bool) -> tuple[str, str]:
    names = tuple(server_names)
    if not prefixed and len(names) == 1:
        return names[0], tool_name
    if not prefixed:
        return "", tool_name
    for server_name in sorted(names, key=len, reverse=True):
        resolved_server_name = str(server_name)
        prefix = f"{resolved_server_name}_"
        if tool_name.startswith(prefix):
            return resolved_server_name, tool_name.removeprefix(prefix)
    return "", tool_name


def _bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if item is not None and str(item))
    return (str(value),)


def _object_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = "`metadata` must be an object."
        raise TypeError(msg)
    return {str(key): item for key, item in value.items()}


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(str(value))


_TOOL_RISKS: tuple[ToolRisk, ...] = ("read_only", "external_read", "write", "destructive", "network")

__all__ = [
    "LoadedMCPTools",
    "MCPRuntimeConfig",
    "MCPServerRuntimeConfig",
    "aload_mcp_tools",
    "build_mcp_tool_descriptors",
    "load_mcp_config",
    "load_mcp_tools",
]
