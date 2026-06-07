"""Pluggable document parsers for knowledge-base ingestion."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast
from xml.etree import ElementTree as ET

from kyuriagents.runtime.mcp import load_mcp_tools
from kyuriagents.tools.registry import tool_name

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.tools import BaseTool

    from kyuriagents.runtime import AgentRuntimeConfig


@dataclass(frozen=True, kw_only=True)
class ParseRequest:
    """Document parse request.

    Args:
        file_path: Local path to the uploaded source file.
        source_uri: Stable URI stored in RAG metadata.
        filename: Original file name supplied by the user.
        mime_type: Source MIME type.
        metadata: Additional parser-specific metadata.
    """

    file_path: Path
    source_uri: str
    filename: str
    mime_type: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ParsedSection:
    """One text span extracted from a document."""

    text: str
    title: str = ""
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True, kw_only=True)
class ParsedDocument:
    """Parsed document returned by a local or remote parser."""

    title: str
    sections: tuple[ParsedSection, ...]
    language: str = "unknown"
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Return all extracted text joined for hashing and chunking."""
        return "\n\n".join(section.text for section in self.sections if section.text.strip())


class DocumentParser(Protocol):
    """Parser contract shared by local and MCP-backed ingestion."""

    name: str
    version: str

    def supports(self, request: ParseRequest) -> bool:
        """Return whether this parser can handle the request."""
        ...

    def parse(self, request: ParseRequest) -> ParsedDocument:
        """Parse a document into text sections."""
        ...


class _InvokableTool(Protocol):
    def invoke(self, input_data: object) -> object:
        """Invoke a parser tool."""
        ...


class LocalPlainTextParser:
    """Extract text from plain text documents."""

    name = "local_plain_text"
    version = "local_plain_text:v1"

    def supports(self, request: ParseRequest) -> bool:
        """Return whether the file looks like a plain text document."""
        suffix = request.filename.lower()
        return request.mime_type.startswith("text/") or suffix.endswith((".txt", ".text"))

    def parse(self, request: ParseRequest) -> ParsedDocument:
        """Decode a plain text document.

        Args:
            request: Source document request.

        Returns:
            Parsed document with one section containing the decoded text.

        Raises:
            ValueError: If the file is empty or cannot be decoded.
        """
        text, encoding = _decode_text(request.file_path.read_bytes())
        text = _normalize_text(text).strip()
        if not text:
            msg = "No text was found in the plain text document."
            raise ValueError(msg)
        return ParsedDocument(title=request.filename, sections=(ParsedSection(text=text),), metadata={"encoding": encoding})


class LocalDocxTextParser:
    """Extract text from modern Word `.docx` documents."""

    name = "local_docx_text"
    version = "local_docx_text:v1"

    def supports(self, request: ParseRequest) -> bool:
        """Return whether the file looks like a Word `.docx` document."""
        return (
            request.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or request.filename.lower().endswith(".docx")
        )

    def parse(self, request: ParseRequest) -> ParsedDocument:
        """Extract paragraph text from a Word `.docx` file.

        Args:
            request: Source document request.

        Returns:
            Parsed document with one section per non-empty paragraph.

        Raises:
            ValueError: If the file is not a readable `.docx` or contains no text.
        """
        try:
            with zipfile.ZipFile(request.file_path) as archive:
                document_xml = archive.read("word/document.xml")
                title = _docx_title(archive, fallback=request.filename)
        except (KeyError, zipfile.BadZipFile) as exc:
            msg = "The Word document could not be read. Local parsing supports `.docx` files only."
            raise ValueError(msg) from exc
        sections = _docx_sections(document_xml)
        if not sections:
            msg = "No extractable text was found in the Word document."
            raise ValueError(msg)
        return ParsedDocument(title=title, sections=sections, metadata={"paragraph_count": len(sections)})


class LocalPdfTextParser:
    """Extract text from ordinary text PDFs using `pypdf`."""

    name = "local_pdf_text"
    version = "local_pdf_text:v1"

    def supports(self, request: ParseRequest) -> bool:
        """Return whether the file looks like a PDF."""
        return request.mime_type == "application/pdf" or request.filename.lower().endswith(".pdf")

    def parse(self, request: ParseRequest) -> ParsedDocument:
        """Extract page text from a PDF.

        Args:
            request: Source document request.

        Returns:
            Parsed document with one section per page containing text.

        Raises:
            ImportError: If `pypdf` is not installed.
            ValueError: If no text could be extracted.
        """
        try:
            reader_cls = import_module("pypdf").PdfReader
        except ImportError as exc:
            msg = "Install `pypdf` or use `DEEPAGENTS_INGESTION_PARSER=mcp` for PDF parsing."
            raise ImportError(msg) from exc

        reader = reader_cls(str(request.file_path))
        sections: list[ParsedSection] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(ParsedSection(text=text, page_start=index, page_end=index))
        if not sections:
            msg = "No extractable text was found in the PDF. OCR is not enabled for the local parser."
            raise ValueError(msg)
        title = _title_from_metadata(reader.metadata, fallback=request.filename)
        return ParsedDocument(title=title, sections=tuple(sections), metadata={"page_count": len(reader.pages)})


class LocalDocumentParser:
    """Parse locally supported document formats."""

    name = "local_document_parser"
    version = "local_document_parser:v1"

    def __init__(self, parsers: Sequence[DocumentParser] | None = None) -> None:
        """Initialize the composite local parser.

        Args:
            parsers: Optional parser sequence for tests or customization.
        """
        self._parsers = tuple(parsers or (LocalPdfTextParser(), LocalDocxTextParser(), LocalPlainTextParser()))

    def supports(self, request: ParseRequest) -> bool:
        """Return whether any local parser supports the request."""
        return any(parser.supports(request) for parser in self._parsers)

    def parse(self, request: ParseRequest) -> ParsedDocument:
        """Parse a document with the first matching local parser."""
        for parser in self._parsers:
            if parser.supports(request):
                return parser.parse(request)
        msg = "Local parser supports PDF, DOCX, and TXT documents."
        raise ValueError(msg)


class MCPDocumentParser:
    """Parse documents by calling a configured MCP tool."""

    name = "mcp_document_parser"
    version = "mcp_document_parser:v1"

    def __init__(
        self,
        *,
        config: AgentRuntimeConfig | None = None,
        tool_name_override: str = "parse_document",
        tools: Sequence[_InvokableTool] | None = None,
    ) -> None:
        """Initialize an MCP parser.

        Args:
            config: Runtime config used to load MCP tools when `tools` is omitted.
            tool_name_override: Name of the MCP tool to invoke.
            tools: Preloaded tools for tests or custom embedding.
        """
        self._config = config
        self._tool_name = tool_name_override
        self._tools: tuple[_InvokableTool, ...] | None = tuple(tools) if tools is not None else None

    def supports(self, request: ParseRequest) -> bool:
        """Return whether the MCP parser should be offered for this request."""
        return bool(request.file_path and request.filename)

    def parse(self, request: ParseRequest) -> ParsedDocument:
        """Call an MCP tool and normalize its result.

        Args:
            request: Source document request.

        Returns:
            Parsed document.

        Raises:
            ValueError: If the configured MCP tool is missing or returns an invalid payload.
        """
        tool = self._resolve_tool()
        payload = {
            "file_path": str(request.file_path),
            "source_uri": request.source_uri,
            "filename": request.filename,
            "mime_type": request.mime_type,
            "metadata": dict(request.metadata),
        }
        result = tool.invoke(payload)
        return _parsed_document_from_mcp_result(result, fallback_title=request.filename)

    def _resolve_tool(self) -> _InvokableTool:
        tools = self._tools
        if tools is None:
            if self._config is None:
                msg = "MCP parser requires either runtime config or preloaded tools."
                raise ValueError(msg)
            loaded = load_mcp_tools(self._config)
            tools = cast("tuple[_InvokableTool, ...]", tuple(loaded.tools))
            self._tools = tools
        for tool in tools:
            name = tool_name(cast("BaseTool", tool))
            if name == self._tool_name or name.endswith(f"_{self._tool_name}"):
                return tool
        names = ", ".join(tool_name(cast("BaseTool", tool)) for tool in tools) or "<none>"
        msg = f"MCP document parser tool `{self._tool_name}` was not found. Available tools: {names}."
        raise ValueError(msg)


class AutoDocumentParser:
    """Try a primary parser and fall back to a secondary parser when available."""

    name = "auto_document_parser"
    version = "auto_document_parser:v1"

    def __init__(self, *, primary: DocumentParser | None, fallback: DocumentParser) -> None:
        """Initialize the parser.

        Args:
            primary: Preferred parser, usually MCP.
            fallback: Local fallback parser.
        """
        self._primary = primary
        self._fallback = fallback

    def supports(self, request: ParseRequest) -> bool:
        """Return whether either parser supports the request."""
        return (self._primary is not None and self._primary.supports(request)) or self._fallback.supports(request)

    def parse(self, request: ParseRequest) -> ParsedDocument:
        """Parse with the primary parser, falling back locally on failure."""
        if self._primary is not None and self._primary.supports(request):
            try:
                return self._primary.parse(request)
            except Exception:
                if not self._fallback.supports(request):
                    raise
        return self._fallback.parse(request)


def build_document_parser(config: AgentRuntimeConfig) -> DocumentParser:
    """Build the configured document parser.

    Args:
        config: Runtime configuration.

    Returns:
        Parser selected by `DEEPAGENTS_INGESTION_PARSER`.
    """
    local = LocalDocumentParser()
    if config.ingestion_parser_mode == "local":
        return local

    mcp_config_path = config.ingestion_mcp_config_path or config.mcp_config_path
    if config.ingestion_parser_mode == "mcp":
        if not mcp_config_path:
            msg = "Set `DEEPAGENTS_INGESTION_MCP_CONFIG_PATH` before using the MCP ingestion parser."
            raise ValueError(msg)
        return MCPDocumentParser(
            config=_config_with_ingestion_mcp_path(config, mcp_config_path),
            tool_name_override=config.ingestion_mcp_tool_name,
        )

    primary = None
    if mcp_config_path:
        primary = MCPDocumentParser(
            config=_config_with_ingestion_mcp_path(config, mcp_config_path),
            tool_name_override=config.ingestion_mcp_tool_name,
        )
    return AutoDocumentParser(primary=primary, fallback=local)


def _config_with_ingestion_mcp_path(config: AgentRuntimeConfig, path: str) -> AgentRuntimeConfig:
    from dataclasses import replace  # noqa: PLC0415

    return replace(config, mcp_config_path=path)


def _parsed_document_from_mcp_result(result: object, *, fallback_title: str) -> ParsedDocument:
    payload = _mcp_payload(result)
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            msg = "MCP parser returned empty text."
            raise ValueError(msg)
        return ParsedDocument(title=fallback_title, sections=(ParsedSection(text=text),))
    if not isinstance(payload, Mapping):
        msg = "MCP parser must return text or an object payload."
        raise TypeError(msg)
    payload_mapping = cast("Mapping[str, object]", payload)
    sections = _sections_from_payload(payload_mapping)
    if not sections:
        text = str(payload_mapping.get("text", "")).strip()
        if text:
            sections = (ParsedSection(text=text),)
    if not sections:
        msg = "MCP parser returned no text sections."
        raise ValueError(msg)
    return ParsedDocument(
        title=str(payload_mapping.get("title") or fallback_title),
        sections=sections,
        language=str(payload_mapping.get("language") or "unknown"),
        metadata=cast("Mapping[str, object]", payload_mapping.get("metadata") or {}),
    )


def _mcp_payload(result: object) -> object:
    content = getattr(result, "content", None)
    if content is not None:
        result = content
    if isinstance(result, str):
        stripped = result.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return result
        return result
    return result


def _sections_from_payload(payload: Mapping[str, object]) -> tuple[ParsedSection, ...]:
    raw = payload.get("sections")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return ()
    sections: list[ParsedSection] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text:
                sections.append(ParsedSection(text=text))
            continue
        if not isinstance(item, Mapping):
            continue
        item_mapping = cast("Mapping[str, object]", item)
        text = str(item_mapping.get("text", "")).strip()
        if not text:
            continue
        sections.append(
            ParsedSection(
                text=text,
                title=str(item_mapping.get("title", "")),
                page_start=_optional_int(item_mapping.get("page_start")),
                page_end=_optional_int(item_mapping.get("page_end")),
            )
        )
    return tuple(sections)


def _decode_text(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    msg = "Plain text document could not be decoded with supported encodings."
    raise ValueError(msg)


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


_WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DUBLIN_CORE_NAMESPACE = "{http://purl.org/dc/elements/1.1/}"


def _docx_sections(document_xml: bytes) -> tuple[ParsedSection, ...]:
    root = _safe_xml_fromstring(document_xml)
    sections: list[ParsedSection] = []
    for paragraph in root.iter(f"{_WORD_NAMESPACE}p"):
        text = _docx_paragraph_text(paragraph).strip()
        if text:
            sections.append(ParsedSection(text=text))
    return tuple(sections)


def _docx_paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{_WORD_NAMESPACE}t":
            parts.append(node.text or "")
        elif node.tag == f"{_WORD_NAMESPACE}tab":
            parts.append("\t")
        elif node.tag == f"{_WORD_NAMESPACE}br":
            parts.append("\n")
    return "".join(parts)


def _docx_title(archive: zipfile.ZipFile, *, fallback: str) -> str:
    try:
        core_xml = archive.read("docProps/core.xml")
    except KeyError:
        return fallback
    root = _safe_xml_fromstring(core_xml)
    title = root.findtext(f"{_DUBLIN_CORE_NAMESPACE}title")
    if title and title.strip():
        return title.strip()
    return fallback


def _safe_xml_fromstring(data: bytes) -> ET.Element:
    lowered = data.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        msg = "DOCX XML with DTD or entity declarations is not supported."
        raise ValueError(msg)
    return ET.fromstring(data)  # noqa: S314  # DOCX XML is pre-screened above and external entities are rejected.


def _title_from_metadata(metadata: object, *, fallback: str) -> str:
    title = getattr(metadata, "title", None)
    if title:
        return str(title)
    return fallback


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(str(value))
