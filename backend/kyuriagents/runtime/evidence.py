"""Structured evidence returned by information subagents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceSource(BaseModel):
    """One source used by an information subagent."""

    title: str = Field(default="", description="Human-readable source title.")
    url: str = Field(default="", description="Source URL or document URI.")
    source_type: Literal["knowledge_base", "web", "memory", "other"] = Field(default="other", description="Source category.")
    quote: str = Field(default="", description="Short supporting excerpt, not a full page.")


class EvidenceFinding(BaseModel):
    """One concise finding with source references."""

    claim: str = Field(description="Atomic claim or observation.")
    source_indices: list[int] = Field(default_factory=list, description="Indices into `sources` supporting the claim.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence from 0 to 1.")


class EvidencePackage(BaseModel):
    """Evidence package returned to the main agent."""

    conclusion: str = Field(description="Short synthesized conclusion for the delegated task.")
    findings: list[EvidenceFinding] = Field(default_factory=list, description="Concise evidence-backed findings.")
    sources: list[EvidenceSource] = Field(default_factory=list, description="Deduplicated sources used by the subagent.")
    missing: list[str] = Field(default_factory=list, description="Information the subagent could not verify.")
    failures: list[str] = Field(default_factory=list, description="Tool failures, blocked pages, or retrieval gaps.")


__all__ = ["EvidenceFinding", "EvidencePackage", "EvidenceSource"]
