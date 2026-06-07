"""Policy checks for runtime tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field

from kyuriagents.tools.types import ToolDescriptor, ToolPolicyDecision, ToolRisk

DEFAULT_ALLOWED_RISKS: frozenset[ToolRisk] = frozenset(("read_only", "external_read", "write"))
"""Conservative default: ordinary writes are allowed; risky writes opt into confirmation."""

DEFAULT_CONFIRMATION_RISKS: frozenset[ToolRisk] = frozenset(("destructive", "network"))
"""Risks that should wait for a human approval flow unless explicitly allowed."""


@dataclass(frozen=True, kw_only=True)
class ToolPolicy:
    """Policy for allowing or blocking tool calls.

    Args:
        allowed_risks: Risk classes allowed to execute.
        confirmation_risks: Risk classes treated as confirmation-gated.
        allow_requires_confirmation: Whether confirmation-gated tools may run.
        allowed_tools: Optional allow-list. When non-empty, all other tools are
            blocked before risk checks.
        denied_tools: Explicit deny-list.
    """

    allowed_risks: frozenset[ToolRisk] = field(default_factory=lambda: DEFAULT_ALLOWED_RISKS)
    confirmation_risks: frozenset[ToolRisk] = field(default_factory=lambda: DEFAULT_CONFIRMATION_RISKS)
    allow_requires_confirmation: bool = False
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    denied_tools: frozenset[str] = field(default_factory=frozenset)

    def evaluate(self, descriptor: ToolDescriptor) -> ToolPolicyDecision:
        """Evaluate one tool descriptor.

        Args:
            descriptor: Descriptor for the requested tool.

        Returns:
            Allow or block decision.
        """
        if descriptor.name in self.denied_tools:
            return ToolPolicyDecision(allowed=False, reason=f"Tool `{descriptor.name}` is denied by policy.")
        if self.allowed_tools and descriptor.name not in self.allowed_tools:
            return ToolPolicyDecision(allowed=False, reason=f"Tool `{descriptor.name}` is not in the allowed tool list.")
        if descriptor.risk not in self.allowed_risks:
            return ToolPolicyDecision(allowed=False, reason=f"Tool `{descriptor.name}` has disallowed risk `{descriptor.risk}`.")
        if not self.allow_requires_confirmation and (descriptor.requires_confirmation or descriptor.risk in self.confirmation_risks):
            return ToolPolicyDecision(allowed=False, reason=f"Tool `{descriptor.name}` requires human confirmation before execution.")
        return ToolPolicyDecision(allowed=True)


def parse_tool_risks(value: str | None, *, default: frozenset[ToolRisk]) -> frozenset[ToolRisk]:
    """Parse comma-separated risk classes.

    Args:
        value: Comma-separated risk classes.
        default: Fallback when `value` is empty.

    Returns:
        Parsed risk set.

    Raises:
        ValueError: If an unknown risk is supplied.
    """
    if not value:
        return default
    valid = set(_TOOL_RISKS)
    risks: set[ToolRisk] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in valid:
            allowed = ", ".join(_TOOL_RISKS)
            msg = f"`ToolRisk` must be one of: {allowed}."
            raise ValueError(msg)
        risks.add(item)  # ty: ignore[invalid-argument-type]  # Guard above narrows to known literal values.
    return frozenset(risks) or default


def parse_tool_names(value: str | None) -> frozenset[str]:
    """Parse comma-separated tool names.

    Args:
        value: Comma-separated tool names.

    Returns:
        Parsed name set.
    """
    if not value:
        return frozenset()
    return frozenset(part.strip() for part in value.split(",") if part.strip())


_TOOL_RISKS: tuple[ToolRisk, ...] = ("read_only", "external_read", "write", "destructive", "network")

__all__ = [
    "DEFAULT_ALLOWED_RISKS",
    "DEFAULT_CONFIRMATION_RISKS",
    "ToolPolicy",
    "parse_tool_names",
    "parse_tool_risks",
]
