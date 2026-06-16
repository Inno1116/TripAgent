"""Task-mode runtime with intent routing, planning, execution, and observation."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage

from kyuriagents.middleware.retrieval import RuntimeContextDefaults, format_rag_context
from kyuriagents.profile import TravelProfileService
from kyuriagents.rag import (
    DashScopeTextReranker,
    ElasticsearchKeywordStore,
    HybridRAGRetriever,
    MilvusVectorStore,
    PostgresChunkTextHydrator,
    RetrievalScope,
)
from kyuriagents.runtime.time_context import current_time_context, format_time_context_block
from kyuriagents.tasks.evidence import EvidenceAgent, EvidenceRequest, create_evidence_agents, format_evidence_package
from kyuriagents.tasks.store import InMemoryTaskStore, PostgresTaskStore, TaskStore, new_step_record
from kyuriagents.tasks.types import (
    PlannedStep,
    StepObservation,
    TaskContext,
    TaskEventRecord,
    TaskIntent,
    TaskPlan,
    TaskRecord,
    TaskStepKind,
    TaskStepRecord,
    ValidationResult,
)
from kyuriagents.tools import ToolCallRecord, ToolCallStatus, ToolDescriptor, ToolRegistry, ToolRisk, default_tool_registry

if TYPE_CHECKING:
    from kyuriagents.runtime import AgentRuntimeConfig
    from kyuriagents.server.identity import MessageRecord
    from kyuriagents.tools.audit import ToolAuditSink

_LOGGER = logging.getLogger(__name__)
_MAX_STEPS = 8
_AUDIT_SUMMARY_LIMIT = 2_000
_TOOL_THREAD_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="kyuriagents-task-tool")
_TASK_HINT_RE = re.compile(
    "(?i)(help me|research|analyze|summarize|compare|draft|create|plan|investigate|"
    "\\u6574\\u7406|\\u8c03\\u7814|\\u5206\\u6790|\\u603b\\u7ed3|\\u751f\\u6210|\\u5b8c\\u6210|\\u8ba1\\u5212|\\u67e5\\u627e)"
)
_RAG_HINT_RE = re.compile(
    "(?i)(knowledge base|document|pdf|paper|file|\\u8d44\\u6599|\\u77e5\\u8bc6\\u5e93|\\u6587\\u6863|\\u8bba\\u6587|\\u6587\\u4ef6|\\u6839\\u636e)"
)
_WEB_HINT_RE = re.compile(
    "(?i)(web|online|internet|current|latest|news|search|website|url|"
    "\\u8054\\u7f51|\\u7f51\\u9875|\\u7f51\\u4e0a|\\u641c\\u7d22|\\u6700\\u65b0|\\u4eca\\u5929|\\u65b0\\u95fb|\\u7f51\\u5740|\\u94fe\\u63a5)"
)
_TRAVEL_HINT_RE = re.compile(
    "(?i)(travel|trip|itinerary|hotel|weather|route|attraction|restaurant|tour|"
    "\\u65c5\\u884c|\\u65c5\\u6e38|\\u884c\\u7a0b|\\u51fa\\u884c|\\u9152\\u5e97|\\u5929\\u6c14|"
    "\\u666f\\u70b9|\\u9910\\u5385|\\u7f8e\\u98df|\\u8def\\u7ebf|\\u4ea4\\u901a|\\u9884\\u7b97)"
)
_PRESEARCH_HINT_RE = re.compile(
    "(?i)(travel|trip|itinerary|recommend|current|latest|where|nearby|"
    "\\u65c5\\u884c|\\u65c5\\u6e38|\\u884c\\u7a0b|\\u51fa\\u884c|\\u63a8\\u8350|\\u5c0f\\u4f17|"
    "\\u653b\\u7565|\\u503c\\u5f97|\\u53bb\\u54ea|\\u666f\\u70b9|\\u9910\\u5385|\\u9152\\u5e97|"
    "\\u5929\\u6c14|\\u8def\\u7ebf|\\u4ea4\\u901a|\\u6700\\u65b0|\\u5f53\\u524d|\\u73b0\\u5728)"
)
_PRESEARCH_MAX_RESULTS = 6
_PRESEARCH_SNIPPET_CHARS = 360


class _Model(Protocol):
    def invoke(self, input_data: object) -> object:
        """Invoke a chat model."""
        ...


ModelFactory = Callable[[], _Model]
ToolHandler = Callable[["TaskExecutionContext", Mapping[str, object]], str]


@dataclass(frozen=True, kw_only=True)
class TaskRuntimeLimits:
    """Hard safety limits for task-mode planning and execution."""

    max_plan_steps: int = 8
    max_replans: int = 2
    max_total_steps: int = 12
    max_tool_calls: int = 8
    max_step_retries: int = 1
    max_runtime_seconds: float = 600.0
    max_same_error: int = 2
    max_step_output_chars: int = 6_000
    tool_timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        """Validate limit values."""
        if self.max_plan_steps <= 0 or self.max_total_steps <= 0:
            msg = "Task step limits must be positive."
            raise ValueError(msg)
        if self.max_plan_steps > self.max_total_steps:
            msg = "`max_plan_steps` must not exceed `max_total_steps`."
            raise ValueError(msg)
        if self.max_step_retries < 0 or self.max_replans < 0:
            msg = "Retry and replan limits must not be negative."
            raise ValueError(msg)
        if self.max_tool_calls <= 0 or self.max_runtime_seconds <= 0 or self.tool_timeout_seconds <= 0:
            msg = "Runtime and tool limits must be positive."
            raise ValueError(msg)
        if self.max_same_error <= 0 or self.max_step_output_chars <= 0:
            msg = "Error and output limits must be positive."
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class TaskExecutionContext:
    """Runtime context supplied to task tool handlers."""

    tenant_id: str
    user_id: str
    thread_id: str
    goal: str
    defaults: RuntimeContextDefaults


@dataclass(frozen=True, kw_only=True)
class TaskRunResult:
    """Result returned by a completed task run."""

    task: TaskRecord
    steps: tuple[TaskStepRecord, ...]
    events: tuple[TaskEventRecord, ...]
    final_answer: str


class IntentRouter:
    """Route a user input into chat, task, retrieval, or clarification intent."""

    def route(self, goal: str, *, forced_intent: TaskIntent | None = None) -> TaskIntent:
        """Route a goal.

        Args:
            goal: User goal text.
            forced_intent: Optional caller override.

        Returns:
            Routed intent.
        """
        if forced_intent is not None:
            return forced_intent
        text = goal.strip()
        if not text:
            return "clarify"
        if _TASK_HINT_RE.search(text):
            return "task"
        if _RAG_HINT_RE.search(text):
            return "rag_query"
        return "chat"


class ContextBuilder:
    """Build compact structured context for planning."""

    def __init__(self, *, tool_registry: ToolRegistry | None = None, profile_service: TravelProfileService | None = None) -> None:
        """Initialize the builder.

        Args:
            tool_registry: Tool metadata source.
            profile_service: Optional structured traveler profile source.
        """
        self._tool_registry = tool_registry or default_tool_registry()
        self._profile_service = profile_service

    def build(
        self,
        *,
        goal: str,
        intent: TaskIntent,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        messages: Sequence[MessageRecord] = (),
        tool_descriptors: Sequence[ToolDescriptor] | None = None,
        constraints: Mapping[str, object] | None = None,
    ) -> TaskContext:
        """Build planner context."""
        descriptors = tuple(self._tool_registry.descriptors.values()) if tool_descriptors is None else tuple(tool_descriptors)
        runtime_time = current_time_context()
        traveler_profile = self._traveler_profile(tenant_id=tenant_id, user_id=user_id)
        return TaskContext(
            goal=goal,
            intent=intent,
            tenant_id=tenant_id,
            user_id=user_id,
            thread_id=thread_id,
            recent_messages=_recent_messages(messages),
            available_tools=tuple(_tool_dict(descriptor) for descriptor in descriptors),
            constraints={
                "language": "zh-CN",
                "max_steps": _MAX_STEPS,
                "current_date": runtime_time["current_date"],
                "current_datetime": runtime_time["current_datetime"],
                "current_year": runtime_time["current_year"],
                "timezone": runtime_time["timezone"],
                "weekday": runtime_time["weekday"],
                "traveler_profile": traveler_profile,
                **dict(constraints or {}),
            },
        )

    def _traveler_profile(self, *, tenant_id: str, user_id: str) -> dict[str, object]:
        if self._profile_service is None:
            return {}
        try:
            record = self._profile_service.get_profile(tenant_id=tenant_id, user_id=user_id)
        except Exception as exc:  # noqa: BLE001  # profile personalization should not block task planning
            _LOGGER.warning("Failed to load traveler profile for task context: %s", exc)
            return {}
        return {
            "profile_version": record.profile_version,
            "profile_data": record.profile_data,
        }


class TaskPreSearcher:
    """Lightweight web-search enrichment before planning.

    This stage discovers real-world candidates for the planner without opening
    pages or turning the result into final evidence.
    """

    def __init__(
        self,
        *,
        config: AgentRuntimeConfig | None = None,
        enabled: bool = False,
        max_results: int = _PRESEARCH_MAX_RESULTS,
    ) -> None:
        """Initialize the pre-searcher."""
        self._config = config
        self._enabled = enabled
        self._max_results = max(1, max_results)
        self._service: object | None = None

    @classmethod
    def from_config(cls, config: AgentRuntimeConfig) -> TaskPreSearcher:
        """Create a pre-searcher from runtime config."""
        return cls(config=config, enabled=bool(config.enable_web_search), max_results=min(_PRESEARCH_MAX_RESULTS, config.web_search_max_results))

    def enrich(self, context: TaskContext) -> TaskContext:
        """Attach a compact pre-search context to planner metadata when useful."""
        if not self._should_search(context):
            return context
        payload = self._presearch(context)
        return _replace_task_context(context, metadata={**context.metadata, "presearch_context": payload})

    def _should_search(self, context: TaskContext) -> bool:
        if not self._enabled or self._config is None:
            return False
        if context.intent != "task":
            return False
        if "presearch_context" in context.metadata:
            return False
        tool_names = {str(tool.get("name")) for tool in context.available_tools}
        if not ({"web_agent", "web_search"} & tool_names):
            return False
        return bool(_PRESEARCH_HINT_RE.search(context.goal) or _TRAVEL_HINT_RE.search(context.goal) or _WEB_HINT_RE.search(context.goal))

    def _presearch(self, context: TaskContext) -> dict[str, object]:
        from kyuriagents.websearch import WebSearchService, blocked_query_reason  # noqa: PLC0415

        query = context.goal.strip()
        payload: dict[str, object] = {
            "enabled": True,
            "query": query,
            "summary": "",
            "candidates": [],
            "missing": [],
            "failures": [],
            "diagnostics": {},
        }
        reason = blocked_query_reason(query)
        if reason:
            payload["failures"] = [reason]
            payload["summary"] = "Pre-search was blocked by policy, so the planner should continue without web seeds."
            return payload
        try:
            service = self._service
            if service is None:
                service = WebSearchService(self._config)
                self._service = service
            response = service.search_with_diagnostics(query, max_results=self._max_results)
        except Exception as exc:  # noqa: BLE001  # pre-search is optional and must never block planning
            payload["failures"] = [str(exc)]
            payload["summary"] = "Pre-search failed, so the planner should create explicit discovery steps if needed."
            return payload

        candidates = [_presearch_candidate(result) for result in response.results[: self._max_results]]
        payload["summary"] = (
            f"Pre-search found {len(candidates)} real-world candidate(s) for planning."
            if candidates
            else "Pre-search found no usable web candidates; plan explicit discovery steps if needed."
        )
        payload["candidates"] = candidates
        payload["missing"] = ["Verify high-value candidates with web/AMap steps before final recommendations."] if candidates else []
        payload["failures"] = list(response.failures)
        payload["diagnostics"] = {
            "planned_queries": list(response.planned_queries),
            "raw_results": response.raw_result_count,
            "deduped": response.deduped_count,
            "filtered": response.filtered_count,
            "cache_hit": response.cache_hit,
        }
        return payload


class LLMPlanner:
    """Generate structured plans with an LLM and deterministic fallback."""

    def __init__(self, *, model_factory: ModelFactory | None = None) -> None:
        """Initialize the planner.

        Args:
            model_factory: Lazy chat model factory.
        """
        self._model_factory = model_factory

    def plan(self, context: TaskContext) -> TaskPlan:
        """Generate a task plan.

        Args:
            context: Structured planner context.

        Returns:
            Valid-looking plan, with semantic validation handled separately.
        """
        if self._model_factory is not None:
            try:
                payload = self._model_plan(context)
                plan = _plan_from_payload(payload, fallback_goal=context.goal)
                if plan.steps:
                    return plan
            except Exception as exc:  # noqa: BLE001  # planning falls back to deterministic heuristics
                _LOGGER.info("Task planner fell back to heuristic plan: %s", exc)
        return heuristic_plan(context)

    def _model_plan(self, context: TaskContext) -> Mapping[str, object]:
        if self._model_factory is None:
            msg = "Planner model factory is not configured."
            raise ValueError(msg)
        model = self._model_factory()
        presearch_context = context.metadata.get("presearch_context")
        hitl_context = context.metadata.get("hitl")
        original_goal = context.metadata.get("original_goal")
        planning_metadata = {
            "original_goal": original_goal,
        }
        prompt = (
            "You are the planner for Kyuriagents city-culture and travel task mode. Return only JSON.\n\n"
            "Planning priority:\n"
            "1. CURRENT_TASK_GOAL is authoritative. Plan only for this goal.\n"
            "2. Do not continue, repeat, or optimize prior tasks unless CURRENT_TASK_GOAL explicitly asks for it.\n"
            "3. BACKGROUND_RECENT_MESSAGES may only resolve explicit references or transferable constraints.\n"
            "4. If CURRENT_TASK_GOAL mentions a new city, destination, attraction, topic, or intent, "
            "do not reuse old destination-specific details from recent messages.\n"
            "5. Transferable background constraints include traveler count/profile, trip length, budget range, "
            "pace, interests, accessibility/dietary constraints, and service needs.\n"
            "6. Non-transferable details include previous city, previous POIs, hotels, restaurants, routes, "
            "weather, local transport, and day-by-day itinerary items.\n"
            "7. RUNTIME_CONSTRAINTS contains runtime date/time and execution limits. Never infer current dates "
            "from model training data.\n"
            "8. CURRENT_TASK_PRESEARCH_SEEDS are fresh web-search candidates generated from CURRENT_TASK_GOAL. "
            "Use them as planning seeds only, not final evidence.\n"
            "9. Verify important pre-search candidates with web/AMap/tool steps before final recommendations.\n"
            "10. AVAILABLE_TOOLS defines the only valid direct tool names for `tool` steps.\n\n"
            "Allowed step kinds: rag, web, tool, process, answer.\n"
            "`rag` searches uploaded/private knowledge-base documents. `web` researches current public web sources.\n"
            "`tool` calls one explicit direct tool from AVAILABLE_TOOLS.\n"
            "`process` is the only intermediate model-analysis step: use it to analyze previous step outputs, "
            "resolve conflicts, compare candidates, derive budget/route conclusions, or organize facts. "
            "It must not fetch new external information.\n"
            "`answer` is only for the final user-facing response.\n"
            "Field distinction:\n"
            "- `kind` is the execution category. It must be exactly one of: rag, web, tool, process, answer.\n"
            "- Tool names from AVAILABLE_TOOLS must never be used as `kind`.\n"
            "- To call any named tool, set `kind` to `tool` and put the exact tool name in `tool_name`.\n"
            '- Invalid: {"kind": "amap_search_poi", "tool_name": "", "input": {"city": "长沙"}}\n'
            '- Valid: {"kind": "tool", "tool_name": "amap_search_poi", "input": {"city": "长沙", "keywords": "景点"}}\n'
            '- Invalid: {"kind": "amap_get_weather", "input": {"city": "长沙"}}\n'
            '- Valid: {"kind": "tool", "tool_name": "amap_get_weather", "input": {"city": "长沙"}}\n'
            "If `tool_name` is non-empty, `kind` MUST be `tool`.\n"
            "For `rag`, `web`, `process`, and `answer`, `tool_name` MUST be empty.\n"
            "For travel planning, create `tool` steps with `tool_name` set to one of these tools when available: "
            "amap_search_poi for places, amap_get_weather for weather, amap_plan_route for route duration, "
            "amap_get_poi_detail for POI details, amap_create_trip_map for map-ready itinerary output, "
            "and estimate_travel_budget for rough budgets.\n"
            "Only use direct tool names from AVAILABLE_TOOLS when kind is `tool`. Prefer rag/web/process before answering.\n"
            "Keep plans short and executable. Do not invent tools.\n"
            "Always include one final answer step as the last step.\n\n"
            f"<CURRENT_TASK_GOAL>\n{context.goal}\n</CURRENT_TASK_GOAL>\n\n"
            f"<BACKGROUND_RECENT_MESSAGES>\n{json.dumps(list(context.recent_messages), ensure_ascii=False)}\n</BACKGROUND_RECENT_MESSAGES>\n\n"
            f"<USER_CLARIFICATIONS>\n{json.dumps(hitl_context if isinstance(hitl_context, Mapping) else {}, ensure_ascii=False)}\n</USER_CLARIFICATIONS>\n\n"
            f"<RUNTIME_CONSTRAINTS>\n{json.dumps(context.constraints, ensure_ascii=False)}\n</RUNTIME_CONSTRAINTS>\n\n"
            f"<CURRENT_TASK_PRESEARCH_SEEDS>\n{json.dumps(presearch_context if isinstance(presearch_context, Mapping) else {}, ensure_ascii=False)}\n</CURRENT_TASK_PRESEARCH_SEEDS>\n\n"
            f"<AVAILABLE_TOOLS>\n{json.dumps(list(context.available_tools), ensure_ascii=False)}\n</AVAILABLE_TOOLS>\n\n"
            f"<PLANNING_METADATA>\n{json.dumps({key: value for key, value in planning_metadata.items() if value}, ensure_ascii=False)}\n</PLANNING_METADATA>\n\n"
            "Return schema:\n"
            '{"goal": string, "summary": string, "steps": ['
            '{"kind": "rag|web|tool|process|answer", "title": string, "instruction": string, '
            '"tool_name": string, "input": object, "depends_on": [string], "parallel_group": string}'
            "]}"
        )
        result = _invoke_model(model, [SystemMessage(content="Return valid JSON only."), HumanMessage(content=prompt)])
        payload = _json_payload(result)
        if not isinstance(payload, Mapping):
            msg = "Planner did not return a JSON object."
            raise TypeError(msg)
        return cast("Mapping[str, object]", payload)


def heuristic_plan(context: TaskContext) -> TaskPlan:
    """Build a conservative fallback plan from intent and tool availability."""
    tools = {str(tool.get("name")) for tool in context.available_tools}
    steps: list[PlannedStep] = []
    if context.intent in {"task", "rag_query"} and "rag_agent" in tools:
        steps.append(
            PlannedStep(
                kind="rag",
                title="Search knowledge-base evidence",
                instruction="Search uploaded and indexed knowledge-base documents, then return a structured evidence package.",
                input={"query": context.goal, "top_k": 6},
                parallel_group="context_lookup",
            )
        )
    elif context.intent in {"task", "rag_query"} and "search_knowledge_base" in tools:
        steps.append(
            PlannedStep(
                kind="tool",
                title="Search knowledge base",
                instruction="Search the user's uploaded and indexed knowledge base for relevant facts.",
                tool_name="search_knowledge_base",
                input={"query": context.goal, "top_k": 6},
                parallel_group="context_lookup",
            )
        )
    if context.intent == "task" and "web_agent" in tools and _WEB_HINT_RE.search(context.goal):
        steps.append(
            PlannedStep(
                kind="web",
                title="Research public web evidence",
                instruction="Search current public web sources and return a structured evidence package with citations.",
                input={"query": context.goal, "max_results": 8, "max_pages": 3},
                parallel_group="context_lookup",
            )
        )
    elif context.intent == "task" and "web_research" in tools and _WEB_HINT_RE.search(context.goal):
        steps.append(
            PlannedStep(
                kind="tool",
                title="Research public web",
                instruction="Search the public web and read relevant pages with sourced excerpts.",
                tool_name="web_research",
                input={"query": context.goal, "max_results": 8, "max_pages": 4},
                parallel_group="context_lookup",
            )
        )
    if context.intent == "task" and _TRAVEL_HINT_RE.search(context.goal):
        if "amap_search_poi" in tools:
            steps.append(
                PlannedStep(
                    kind="tool",
                    title="Search travel places",
                    instruction="Search relevant attractions, restaurants, hotels, or stations for the destination with AMap.",
                    tool_name="amap_search_poi",
                    input={"query": context.goal, "keywords": context.goal, "city": ""},
                )
            )
        if "amap_get_weather" in tools:
            steps.append(
                PlannedStep(
                    kind="tool",
                    title="Check destination weather",
                    instruction="Check destination weather before arranging outdoor-heavy days.",
                    tool_name="amap_get_weather",
                    input={"city": context.goal},
                )
            )
        if "estimate_travel_budget" in tools:
            steps.append(
                PlannedStep(
                    kind="tool",
                    title="Estimate travel budget",
                    instruction="Estimate a rough budget from trip length, travelers, and budget level.",
                    tool_name="estimate_travel_budget",
                    input={"days": 3, "travelers": 1, "budget_level": "medium", "city": context.goal},
                )
            )
    if context.intent == "task":
        steps.append(
            PlannedStep(
                kind="process",
                title="Organize gathered information",
                instruction="Organize tool results and identify which facts support the final answer.",
            )
        )
    steps.append(
        PlannedStep(
            kind="answer",
            title="Generate final answer",
            instruction="Answer the user directly in Chinese, with evidence and any remaining uncertainty.",
        )
    )
    return TaskPlan(goal=context.goal, summary="Heuristic task plan.", steps=tuple(steps[:_MAX_STEPS]))


class PlanValidator:
    """Validate model-generated plans before execution."""

    def validate(self, plan: TaskPlan, context: TaskContext) -> ValidationResult:
        """Validate a task plan."""
        errors: list[str] = []
        max_steps = _int_value(context.constraints.get("max_steps"), default=_MAX_STEPS)
        if not plan.steps:
            errors.append("Plan must contain at least one step.")
        if len(plan.steps) > max_steps:
            errors.append(f"Plan has more than {max_steps} steps.")
        available = {str(tool.get("name")): tool for tool in context.available_tools}
        has_answer = False
        for index, step in enumerate(plan.steps):
            errors.extend(_validate_planned_step(step, index=index, available=available))
            if step.kind == "answer":
                has_answer = True
        if not has_answer:
            errors.append("Plan must end with or include an answer step.")
        return ValidationResult(valid=not errors, errors=tuple(errors))


def _validate_planned_step(step: PlannedStep, *, index: int, available: Mapping[str, Mapping[str, object]]) -> list[str]:
    errors: list[str] = []
    label = f"Step {index + 1}"
    if step.kind not in {"tool", "rag", "web", "process", "answer"}:
        errors.append(f"{label} has unsupported kind `{step.kind}`.")
        return errors
    if step.kind != "tool" and step.tool_name:
        errors.append(f"{label} has `tool_name` but kind is `{step.kind}`; use kind `tool` or clear `tool_name`.")
        return errors
    if step.kind == "rag":
        if "rag_agent" not in available:
            errors.append(f"{label} requires unavailable evidence agent `rag_agent`.")
        return errors
    if step.kind == "web":
        if "web_agent" not in available:
            errors.append(f"{label} requires unavailable evidence agent `web_agent`.")
        return errors
    if step.kind != "tool":
        return errors
    if not step.tool_name:
        errors.append(f"{label} is missing `tool_name`.")
    elif step.tool_name not in available:
        errors.append(f"{label} references unavailable tool `{step.tool_name}`.")
    elif bool(available[step.tool_name].get("requires_confirmation")):
        errors.append(f"{label} requires confirmation, which task mode has not wired yet.")
    return errors


class TaskToolExecutor:
    """Execute task-mode tool steps through registered handlers."""

    def __init__(
        self,
        *,
        handlers: Mapping[str, ToolHandler],
        descriptors: Sequence[ToolDescriptor],
        timeout_seconds: float = 45.0,
        audit_sink: ToolAuditSink | None = None,
    ) -> None:
        """Initialize the executor."""
        self._handlers = dict(handlers)
        self.descriptors = tuple(descriptors)
        self._descriptors = {descriptor.name: descriptor for descriptor in self.descriptors}
        self._timeout_seconds = timeout_seconds
        self._audit_sink = audit_sink

    @classmethod
    def from_config(cls, config: AgentRuntimeConfig) -> TaskToolExecutor:
        """Create default read-only task tools from runtime config."""
        handlers: dict[str, ToolHandler] = {}
        descriptors: list[ToolDescriptor] = []
        if config.enable_rag:
            handlers["search_knowledge_base"] = _rag_handler(config)
            descriptors.append(default_tool_registry().descriptor_for("search_knowledge_base"))
        if config.enable_web_search:
            from kyuriagents.websearch import web_search_tool_descriptors  # noqa: PLC0415

            handlers["web_search"] = _web_search_handler(config)
            handlers["web_research"] = _web_research_handler(config)
            handlers["web_fetch_page"] = _web_fetch_page_handler(config)
            web_descriptors = web_search_tool_descriptors(timeout_seconds=max(1, int(config.web_render_timeout_seconds)))
            descriptors.extend(descriptor for descriptor in web_descriptors if descriptor.name in handlers)
        if config.enable_travel_tools:
            from kyuriagents.travel import create_travel_tool_handlers, travel_tool_descriptors  # noqa: PLC0415

            travel_handlers = create_travel_tool_handlers(config)
            handlers.update(travel_handlers)
            descriptors.extend(
                descriptor
                for descriptor in travel_tool_descriptors(timeout_seconds=max(1, int(config.web_search_timeout_seconds)))
                if descriptor.name in travel_handlers
            )
        audit_sink = None
        if config.enable_tool_audit and config.postgres_dsn:
            from kyuriagents.tools import PostgresToolAuditSink  # noqa: PLC0415

            audit_sink = PostgresToolAuditSink(dsn=config.postgres_dsn)
        return cls(handlers=handlers, descriptors=descriptors, audit_sink=audit_sink)

    def execute(self, step: TaskStepRecord, context: TaskExecutionContext) -> str:
        """Execute one tool step."""
        handler = self._handlers.get(step.tool_name)
        if handler is None:
            msg = f"Task tool `{step.tool_name}` is not available."
            raise ValueError(msg)
        descriptor = self._descriptors.get(step.tool_name)
        started = time.perf_counter()
        future: Future[str] = _TOOL_THREAD_POOL.submit(handler, context, step.input)
        try:
            output = future.result(timeout=self._timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            msg = f"Task tool `{step.tool_name}` timed out after {self._timeout_seconds:.1f}s."
            self._record_audit(step, context=context, descriptor=descriptor, status="error", started=started, error=msg)
            raise TimeoutError(msg) from exc
        except Exception as exc:
            self._record_audit(step, context=context, descriptor=descriptor, status="error", started=started, error=str(exc))
            raise
        self._record_audit(step, context=context, descriptor=descriptor, status="success", started=started, output=output)
        return output

    def _record_audit(
        self,
        step: TaskStepRecord,
        *,
        context: TaskExecutionContext,
        descriptor: ToolDescriptor | None,
        status: ToolCallStatus,
        started: float,
        output: object = None,
        error: str | None = None,
    ) -> None:
        if self._audit_sink is None or descriptor is None:
            return
        record = ToolCallRecord(
            call_id=step.step_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            thread_id=context.thread_id,
            tool_name=descriptor.name,
            source=descriptor.source,
            risk=descriptor.risk,
            status=status,
            input_summary=_json_summary(step.input),
            output_summary=_json_summary(output),
            duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
            error=error,
            created_at=datetime.now(tz=UTC).isoformat(),
            metadata={"task_mode": True, "step_id": step.step_id, "requires_confirmation": descriptor.requires_confirmation, **descriptor.metadata},
        )
        try:
            self._audit_sink.record(record)
        except Exception as exc:  # noqa: BLE001  # Audit failures should not break task execution.
            _LOGGER.warning("Task tool audit failed: %s", exc)


class TaskStepExecutor:
    """Execute information, reasoning, direct-tool, and answer steps."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory | None,
        tool_executor: TaskToolExecutor,
        evidence_agents: Mapping[str, EvidenceAgent] | None = None,
    ) -> None:
        """Initialize the executor."""
        self._model_factory = model_factory
        self._tool_executor = tool_executor
        self._evidence_agents = dict(evidence_agents or {})

    @property
    def tool_descriptors(self) -> tuple[ToolDescriptor, ...]:
        """Return tool descriptors visible to the planner."""
        descriptors = list(self._tool_executor.descriptors)
        if "rag" in self._evidence_agents:
            descriptors = [descriptor for descriptor in descriptors if descriptor.name != "search_knowledge_base"]
        if "web" in self._evidence_agents:
            descriptors = [
                descriptor
                for descriptor in descriptors
                if descriptor.name not in {"web_search", "web_research", "web_fetch_page", "web_fetch_static", "web_render_page"}
            ]
        descriptors.extend(agent.descriptor for agent in self._evidence_agents.values())
        return tuple(descriptors)

    def execute(self, step: TaskStepRecord, *, context: TaskExecutionContext, previous_steps: Sequence[TaskStepRecord]) -> str:
        """Execute one step."""
        if step.kind in {"rag", "web"}:
            return self._evidence_step(step, context=context)
        if step.kind == "tool":
            return self._tool_executor.execute(step, context)
        if step.kind == "process":
            return self._model_step(
                step,
                context=context,
                previous_steps=previous_steps,
                purpose="Analyze gathered evidence, resolve conflicts, and produce concise structured conclusions.",
            )
        if step.kind == "answer":
            return self._model_step(
                step,
                context=context,
                previous_steps=previous_steps,
                purpose="Generate the final user-facing answer in Chinese.",
            )
        msg = f"Unsupported task step kind `{step.kind}`."
        raise ValueError(msg)

    def _evidence_step(self, step: TaskStepRecord, *, context: TaskExecutionContext) -> str:
        agent = self._evidence_agents.get(step.kind)
        if agent is None:
            msg = f"Evidence agent for `{step.kind}` steps is not available."
            raise ValueError(msg)
        query = str(step.input.get("query") or step.instruction or context.goal)
        package = agent.run(
            EvidenceRequest(
                query=query,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                thread_id=context.thread_id,
                goal=context.goal,
                input=step.input,
            )
        )
        return format_evidence_package(package)

    def _model_step(
        self,
        step: TaskStepRecord,
        *,
        context: TaskExecutionContext,
        previous_steps: Sequence[TaskStepRecord],
        purpose: str,
    ) -> str:
        if self._model_factory is None:
            return _fallback_model_step(step, context=context, previous_steps=previous_steps)
        model = self._model_factory()
        prompt = (
            f"{purpose}\n\n"
            f"{format_time_context_block()}\n\n"
            f"User goal: {context.goal}\n"
            f"Current step: {step.title}\n"
            f"Step instruction: {step.instruction}\n\n"
            f"Previous step results:\n{_previous_step_text(previous_steps)}"
        )
        messages = (
            SystemMessage(content="You are Kyuriagents task executor. Be concise and concrete."),
            HumanMessage(content=prompt),
        )
        return _invoke_model(model, list(messages))


class Observer:
    """Observe step outcomes and decide how execution should continue."""

    def observe(
        self,
        step: TaskStepRecord,
        *,
        error: Exception | None = None,
        limits: TaskRuntimeLimits | None = None,
        can_replan: bool = False,
        same_error_exceeded: bool = False,
    ) -> StepObservation:
        """Return the next execution decision."""
        resolved_limits = limits or TaskRuntimeLimits()
        if error is not None:
            if step.attempts <= resolved_limits.max_step_retries and not same_error_exceeded:
                return StepObservation(decision="retry", message=str(error))
            if step.kind == "tool" and can_replan and not same_error_exceeded:
                return StepObservation(decision="replan", message=str(error))
            if _is_external_step(step) and _can_skip_failed_tool(step):
                return StepObservation(decision="skip", message=str(error))
            return StepObservation(decision="fail", message=str(error))
        if step.kind == "answer":
            return StepObservation(decision="finish", message="Task has a final answer.")
        return StepObservation(decision="continue", message="Step succeeded.")


class TaskRuntime:
    """Base task runtime shared by the LangGraph implementation."""

    def __init__(
        self,
        *,
        store: TaskStore | None = None,
        router: IntentRouter | None = None,
        context_builder: ContextBuilder | None = None,
        planner: LLMPlanner | None = None,
        validator: PlanValidator | None = None,
        executor: TaskStepExecutor | None = None,
        observer: Observer | None = None,
        presearcher: TaskPreSearcher | None = None,
        limits: TaskRuntimeLimits | None = None,
    ) -> None:
        """Initialize the runtime."""
        self._limits = limits or TaskRuntimeLimits()
        self.store = store or InMemoryTaskStore()
        self._router = router or IntentRouter()
        self._context_builder = context_builder or ContextBuilder()
        self._planner = planner or LLMPlanner()
        self._validator = validator or PlanValidator()
        self._executor = executor or TaskStepExecutor(
            model_factory=None,
            tool_executor=TaskToolExecutor(handlers={}, descriptors=(), timeout_seconds=self._limits.tool_timeout_seconds),
        )
        self._observer = observer or Observer()
        self._presearcher = presearcher or TaskPreSearcher()

    @classmethod
    def from_config(cls, config: AgentRuntimeConfig) -> TaskRuntime:
        """Create the deployed task runtime."""
        from kyuriagents.tasks.graph_runtime import GraphTaskRuntime  # noqa: PLC0415

        return GraphTaskRuntime.from_config(config)

    def run(
        self,
        *,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        goal: str,
        title: str = "",
        messages: Sequence[MessageRecord] = (),
        forced_intent: TaskIntent | None = "task",
        disabled_tools: Sequence[str] = (),
    ) -> TaskRunResult:
        """Create a task and execute it through the concrete runtime."""
        task = self.store.create_task(tenant_id=tenant_id, user_id=user_id, thread_id=thread_id, goal=goal, title=title, intent="task")
        self.store.add_event(task_id=task.task_id, event_type="created", message="Task created.")
        return self.run_existing_task(task=task, messages=messages, forced_intent=forced_intent, disabled_tools=disabled_tools)

    def _record_presearch_event(self, task: TaskRecord, context: TaskContext) -> None:
        payload = context.metadata.get("presearch_context")
        if not isinstance(payload, Mapping):
            return
        self.store.add_event(task_id=task.task_id, event_type="presearched", message=_presearch_event_message(payload), payload=dict(payload))

    def run_existing_task(
        self,
        *,
        task: TaskRecord,
        messages: Sequence[MessageRecord] = (),
        forced_intent: TaskIntent | None = "task",
        disabled_tools: Sequence[str] = (),
    ) -> TaskRunResult:
        """Run a previously created task.

        Subclasses provide the concrete orchestration strategy. The deployed
        runtime is `GraphTaskRuntime`, which executes this through LangGraph.
        """
        raise NotImplementedError("Use GraphTaskRuntime for task execution.")

    def _skip_step_for_tool_limit(self, *, task: TaskRecord, step: TaskStepRecord) -> TaskStepRecord:
        output = f"Skipped because the task reached the tool-call limit of {self._limits.max_tool_calls}."
        skipped = self.store.update_step(step.step_id, status="skipped", output=output, finished=True)
        self.store.add_event(task_id=task.task_id, step_id=skipped.step_id, event_type="skipped", message=skipped.output)
        return skipped

    def _skip_failed_step(self, *, task: TaskRecord, step: TaskStepRecord, message: str, error: Exception) -> TaskStepRecord:
        output = _truncate_text(f"Skipped after failure: {message}", max_chars=self._limits.max_step_output_chars)
        skipped = self.store.update_step(
            step.step_id,
            status="skipped",
            output=output,
            error_message=str(error),
            finished=True,
        )
        self.store.add_event(task_id=task.task_id, step_id=skipped.step_id, event_type="skipped", message=output)
        return skipped

    def _skip_remaining(self, *, task: TaskRecord, steps: Sequence[TaskStepRecord]) -> None:
        for step in steps:
            if step.status != "pending":
                continue
            skipped = self.store.update_step(
                step.step_id,
                status="skipped",
                output="Skipped because the task was replanned.",
                finished=True,
            )
            self.store.add_event(task_id=task.task_id, step_id=skipped.step_id, event_type="skipped", message=skipped.output)

    def _result(self, task: TaskRecord) -> TaskRunResult:
        return TaskRunResult(
            task=task,
            steps=tuple(self.store.list_steps(task_id=task.task_id)),
            events=tuple(self.store.list_events(task_id=task.task_id)),
            final_answer=task.final_answer,
        )


def _records_from_plan(task_id: str, plan: TaskPlan, *, start_index: int = 0) -> list[TaskStepRecord]:
    records: list[TaskStepRecord] = []
    registry = default_tool_registry()
    for index, step in enumerate(plan.steps):
        descriptor = registry.descriptor_for(step.tool_name) if step.tool_name else None
        risk: ToolRisk = "external_read" if step.kind == "web" else "read_only"
        if descriptor is not None:
            risk = descriptor.risk
        records.append(
            new_step_record(
                task_id=task_id,
                step_index=start_index + index,
                kind=cast("TaskStepKind", step.kind),
                title=step.title,
                instruction=step.instruction,
                tool_name=step.tool_name,
                input=step.input,
                depends_on=step.depends_on,
                parallel_group=step.parallel_group,
                risk=risk,
                requires_confirmation=descriptor.requires_confirmation if descriptor is not None else False,
            )
        )
    return records


def _normalize_plan(plan: TaskPlan, *, limits: TaskRuntimeLimits, max_steps: int | None = None) -> TaskPlan:
    step_limit = max(1, min(limits.max_plan_steps, max_steps or limits.max_plan_steps))
    non_answer = tuple(step for step in plan.steps if step.kind != "answer")
    answer = next((step for step in reversed(plan.steps) if step.kind == "answer"), _default_answer_step())
    return TaskPlan(
        goal=plan.goal,
        summary=plan.summary,
        steps=(*non_answer[: step_limit - 1], answer),
        metadata=plan.metadata,
    )


def _default_answer_step() -> PlannedStep:
    return PlannedStep(
        kind="answer",
        title="Generate final answer",
        instruction="Answer the user directly in Chinese, using completed step outputs and noting uncertainty.",
    )


def _replan_context(
    context: TaskContext,
    *,
    failed_step: TaskStepRecord,
    completed: Sequence[TaskStepRecord],
    error: Exception,
) -> TaskContext:
    disabled_tools = {failed_step.tool_name} if failed_step.tool_name else set()
    available_tools = tuple(tool for tool in context.available_tools if str(tool.get("name")) not in disabled_tools)
    metadata = {
        **context.metadata,
        "replan": True,
        "failed_step": {
            "title": failed_step.title,
            "kind": failed_step.kind,
            "tool_name": failed_step.tool_name,
            "error": str(error),
        },
        "completed_step_outputs": _previous_step_text(completed),
    }
    constraints = {
        **context.constraints,
        "avoid_tools": sorted(disabled_tools),
        "replan_reason": str(error),
    }
    return TaskContext(
        goal=context.goal,
        intent=context.intent,
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        thread_id=context.thread_id,
        recent_messages=context.recent_messages,
        available_tools=available_tools,
        constraints=constraints,
        metadata=metadata,
    )


def _can_skip_failed_tool(step: TaskStepRecord) -> bool:
    return step.risk in {"read_only", "external_read", "network"}


def _is_external_step(step: TaskStepRecord) -> bool:
    return step.kind in {"tool", "rag", "web"}


def _counts_toward_tool_limit(step: TaskStepRecord) -> bool:
    return step.kind in {"tool", "rag", "web"}


def _error_key(error: Exception) -> str:
    return f"{type(error).__name__}:{str(error)[:160]}"


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[truncated after {max_chars} characters]"


def _json_summary(value: object) -> str:
    if value is None:
        return ""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    if len(text) <= _AUDIT_SUMMARY_LIMIT:
        return text
    return f"{text[: _AUDIT_SUMMARY_LIMIT - 14]}...[truncated]"


def _rag_handler(config: AgentRuntimeConfig) -> ToolHandler:
    retriever: HybridRAGRetriever | None = None

    def handler(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        nonlocal retriever
        if retriever is None:
            from kyuriagents.runtime.dashscope import create_dashscope_embed_query  # noqa: PLC0415

            embed_query = create_dashscope_embed_query(config)
            retriever = HybridRAGRetriever(
                vector_searcher=MilvusVectorStore(
                    collection_name=config.rag_milvus_collection,
                    uri=config.rag_milvus_uri,
                    token=config.rag_milvus_token,
                    db_name=config.rag_milvus_db,
                    embed_query=embed_query,
                ),
                keyword_searcher=ElasticsearchKeywordStore(index=config.rag_es_index, url=config.rag_es_url),
                chunk_hydrator=PostgresChunkTextHydrator(dsn=config.postgres_dsn) if config.postgres_dsn else None,
                reranker=DashScopeTextReranker(
                    api_key=config.dashscope_api_key or "",
                    model=config.rag_rerank_model,
                    endpoint=config.rag_rerank_url,
                    timeout_seconds=config.rag_rerank_timeout_seconds,
                )
                if config.rag_rerank_model
                else None,
            )
        query = str(input_data.get("query") or context.goal)
        top_k = _int_value(input_data.get("top_k"), default=6)
        scope = RetrievalScope(tenant_id=context.tenant_id, user_id=context.user_id, kb_ids=config.rag_kb_ids)
        chunks = retriever.retrieve(query, scope=scope, top_k=top_k)
        return format_rag_context(chunks) if chunks else "No relevant knowledge-base chunks found."

    return handler


def _web_search_handler(config: AgentRuntimeConfig) -> ToolHandler:
    service = None

    def handler(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        nonlocal service
        from kyuriagents.websearch import WebSearchService, blocked_query_reason, format_web_search_results  # noqa: PLC0415

        if service is None:
            service = WebSearchService(config)
        query = str(input_data.get("query") or context.goal)
        reason = blocked_query_reason(query)
        if reason:
            return reason
        max_results = _int_value(input_data.get("max_results"), default=config.web_search_max_results)
        return format_web_search_results(service.search(query, max_results=max_results), query=query)

    return handler


def _web_research_handler(config: AgentRuntimeConfig) -> ToolHandler:
    service = None

    def handler(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        nonlocal service
        from kyuriagents.websearch import WebSearchService, blocked_query_reason, format_web_research  # noqa: PLC0415

        if service is None:
            service = WebSearchService(config)
        query = str(input_data.get("query") or context.goal)
        reason = blocked_query_reason(query)
        if reason:
            return reason
        max_results = _int_value(input_data.get("max_results"), default=config.web_search_max_results)
        max_pages = _int_value(input_data.get("max_pages"), default=config.web_fetch_max_pages)
        return format_web_research(service.research(query, max_results=max_results, max_pages=max_pages))

    return handler


def _web_fetch_page_handler(config: AgentRuntimeConfig) -> ToolHandler:
    service = None

    def handler(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        del context
        nonlocal service
        from kyuriagents.websearch import WebSearchService, format_fetched_page  # noqa: PLC0415

        if service is None:
            service = WebSearchService(config)
        url = str(input_data.get("url") or "")
        if not url:
            return "No URL was supplied."
        return format_fetched_page(service.fetch_url(url))

    return handler


def _recent_messages(messages: Sequence[MessageRecord], *, limit: int = 12) -> tuple[dict[str, str], ...]:
    recent = messages[-limit:]
    return tuple({"role": message.role, "content": message.content[:1200]} for message in recent)


def _replace_task_context(context: TaskContext, *, metadata: Mapping[str, object]) -> TaskContext:
    return TaskContext(
        goal=context.goal,
        intent=context.intent,
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        thread_id=context.thread_id,
        recent_messages=context.recent_messages,
        available_tools=context.available_tools,
        constraints=context.constraints,
        metadata=dict(metadata),
    )


def _presearch_candidate(result: object) -> dict[str, object]:
    metadata = getattr(result, "metadata", {}) or {}
    title = str(getattr(result, "title", "") or "")
    url = str(getattr(result, "url", "") or "")
    snippet = _truncate_text(str(getattr(result, "snippet", "") or ""), max_chars=_PRESEARCH_SNIPPET_CHARS)
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "score": getattr(result, "score", None),
        "source": str(getattr(result, "source", "") or "searxng"),
        "planned_query": str(metadata.get("planned_query") or ""),
        "type": _presearch_candidate_type(f"{title}\n{snippet}"),
        "reason": "Search title/snippet matched the current task; verify before using in the final plan.",
    }


def _presearch_candidate_type(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("weather", "forecast", "\u5929\u6c14", "\u9884\u62a5")):
        return "weather_hint"
    if any(token in lowered for token in ("hotel", "\u9152\u5e97", "\u4f4f\u5bbf", "\u6c11\u5bbf")):
        return "hotel"
    if any(token in lowered for token in ("restaurant", "food", "bar", "\u9910\u5385", "\u7f8e\u98df", "\u9152\u5427", "\u5c0f\u9152\u9986")):
        return "restaurant"
    if any(token in lowered for token in ("route", "metro", "train", "flight", "\u8def\u7ebf", "\u5730\u94c1", "\u9ad8\u94c1", "\u822a\u73ed")):
        return "transport"
    if any(token in lowered for token in ("ticket", "opening", "\u95e8\u7968", "\u5f00\u653e", "\u9884\u7ea6")):
        return "availability"
    return "candidate"


def _presearch_event_message(payload: Mapping[str, object]) -> str:
    candidates = payload.get("candidates")
    failures = payload.get("failures")
    candidate_count = len(candidates) if isinstance(candidates, Sequence) and not isinstance(candidates, str | bytes) else 0
    failure_count = len(failures) if isinstance(failures, Sequence) and not isinstance(failures, str | bytes) else 0
    if candidate_count:
        return f"Pre-search found {candidate_count} planning seed(s)."
    if failure_count:
        return "Pre-search failed; planning will continue without web seeds."
    return "Pre-search completed with no usable planning seeds."


def _tool_dict(descriptor: ToolDescriptor) -> dict[str, object]:
    return {
        "name": descriptor.name,
        "description": descriptor.description,
        "risk": descriptor.risk,
        "source": descriptor.source,
        "requires_confirmation": descriptor.requires_confirmation,
        "tags": list(descriptor.tags),
    }


def _filter_descriptors(descriptors: Sequence[ToolDescriptor], *, disabled_tools: Sequence[str]) -> tuple[ToolDescriptor, ...]:
    disabled = set(disabled_tools)
    return tuple(descriptor for descriptor in descriptors if descriptor.name not in disabled)


def _context_payload(context: TaskContext) -> dict[str, object]:
    return {
        "goal": context.goal,
        "current_goal": context.goal,
        "intent": context.intent,
        "recent_messages": list(context.recent_messages),
        "available_tools": list(context.available_tools),
        "constraints": context.constraints,
        "metadata": context.metadata,
    }


def _plan_from_payload(payload: Mapping[str, object], *, fallback_goal: str) -> TaskPlan:
    raw_steps = payload.get("steps")
    steps: list[PlannedStep] = []
    if isinstance(raw_steps, Sequence) and not isinstance(raw_steps, str | bytes):
        for item in raw_steps:
            if not isinstance(item, Mapping):
                continue
            raw = cast("Mapping[str, object]", item)
            steps.append(
                PlannedStep(
                    kind=str(raw.get("kind") or ""),
                    title=str(raw.get("title") or raw.get("kind") or "Step"),
                    instruction=str(raw.get("instruction") or ""),
                    tool_name=str(raw.get("tool_name") or ""),
                    input=_dict_value(raw.get("input")),
                    depends_on=tuple(str(value) for value in _list_value(raw.get("depends_on"))),
                    parallel_group=str(raw.get("parallel_group") or ""),
                )
            )
    return TaskPlan(
        goal=str(payload.get("goal") or fallback_goal),
        summary=str(payload.get("summary") or ""),
        steps=tuple(steps[:_MAX_STEPS]),
        metadata=_dict_value(payload.get("metadata")),
    )


def _invoke_model(model: _Model, messages: object) -> str:
    result = model.invoke(messages)
    content = getattr(result, "content", result)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _json_payload(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def _fallback_model_step(step: TaskStepRecord, *, context: TaskExecutionContext, previous_steps: Sequence[TaskStepRecord]) -> str:
    if step.kind == "answer":
        previous = _previous_step_text(previous_steps)
        if previous:
            return f"Task results organized from current steps:\n\n{previous}"
        return f"Task goal: {context.goal}"
    return _previous_step_text(previous_steps) or step.instruction or step.title


def _previous_step_text(steps: Sequence[TaskStepRecord]) -> str:
    parts = []
    for step in steps:
        output = step.output.strip()
        if output:
            parts.append(f"[{step.title}]\n{output}")
    return "\n\n".join(parts)


def _int_value(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value if value > 0 else default
    if not isinstance(value, str | bytes | bytearray):
        return default
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def _dict_value(value: object) -> dict[str, object]:
    return dict(cast("Mapping[str, object]", value)) if isinstance(value, Mapping) else {}


def _list_value(value: object) -> list[object]:
    return list(value) if isinstance(value, list | tuple) else []


__all__ = [
    "ContextBuilder",
    "IntentRouter",
    "LLMPlanner",
    "Observer",
    "PlanValidator",
    "TaskExecutionContext",
    "TaskRunResult",
    "TaskRuntime",
    "TaskRuntimeLimits",
    "TaskPreSearcher",
    "TaskStepExecutor",
    "TaskToolExecutor",
    "heuristic_plan",
]
