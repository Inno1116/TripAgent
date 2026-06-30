"""LangGraph-backed task planning runtime."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from kyuriagents.middleware.retrieval import RuntimeContextDefaults
from kyuriagents.profile import ProfileCandidate, TravelProfileService, normalize_profile_candidate
from kyuriagents.tasks.evidence import create_evidence_agents
from kyuriagents.tasks.runtime import (
    ContextBuilder,
    IntentRouter,
    LLMPlanner,
    Observer,
    PlanValidator,
    TaskExecutionContext,
    TaskPreSearcher,
    TaskRunResult,
    TaskRuntime,
    TaskRuntimeLimits,
    TaskStepExecutor,
    TaskToolExecutor,
    _can_skip_failed_tool,
    _counts_toward_tool_limit,
    _error_key,
    _filter_descriptors,
    _invoke_model,
    _json_payload,
    _normalize_plan,
    _previous_step_text,
    _records_from_plan,
    _replan_context,
    _truncate_text,
)
from kyuriagents.tasks.store import InMemoryTaskStore, PostgresTaskStore, TaskStore
from kyuriagents.tasks.types import TaskContext, TaskIntent, TaskPlan, TaskRecord, TaskStepRecord

if TYPE_CHECKING:
    from kyuriagents.runtime import AgentRuntimeConfig
    from kyuriagents.server.identity import MessageRecord

_Route = Literal["presearch", "plan", "select_step", "execute_step", "replan", "ask_user", "answer", "fail"]
_TerminalStatus = Literal["pending", "succeeded", "failed"]


class _Model(Protocol):
    def invoke(self, input_data: object) -> object:
        """Invoke a chat model."""
        ...


class _ModelFactory(Protocol):
    def __call__(self) -> _Model:
        """Create a chat model."""
        ...


@dataclass(frozen=True, kw_only=True)
class ClarificationDecision:
    """Structured decision from the clarification node.

    Args:
        need_user_input: Whether execution should pause for human input.
        question: User-facing follow-up question when input is required.
        reason: Short machine-facing reason for observability.
        normalized_goal: Optional normalized goal used by downstream nodes.
        profile_candidates: Explicit user-sourced traveler profile changes that
            may be committed only after successful task completion.
    """

    need_user_input: bool = False
    question: str = ""
    reason: str = ""
    normalized_goal: str = ""
    profile_candidates: tuple[ProfileCandidate, ...] = ()


class ClarificationJudge:
    """Decide whether the task needs human-in-the-loop clarification."""

    def __init__(self, *, model_factory: _ModelFactory | None = None) -> None:
        """Initialize the judge.

        Args:
            model_factory: Optional lazy chat model factory.
        """
        self._model_factory = model_factory

    def judge(self, context: TaskContext) -> ClarificationDecision:
        """Return a clarification decision for a task context.

        Args:
            context: Current task context.

        Returns:
            Structured clarification decision.
        """
        if not context.goal.strip():
            return ClarificationDecision(need_user_input=True, question="Please tell me what task you want to complete.", reason="empty_goal")
        if self._model_factory is None:
            return ClarificationDecision(normalized_goal=context.goal)
        try:
            return _clarification_from_payload(self._model_judge(context), fallback_goal=context.goal)
        except Exception:  # noqa: BLE001  # ambiguity checks should never block task planning
            return ClarificationDecision(normalized_goal=context.goal)

    def _model_judge(self, context: TaskContext) -> Mapping[str, object]:
        factory = self._model_factory
        if factory is None:
            msg = "Clarification model factory is not configured."
            raise ValueError(msg)
        model = factory()
        prompt = (
            "You are the clarification gate for a travel-oriented task planner. Return only JSON.\n"
            "Ask the user only when the current user goal cannot be usefully planned without one short follow-up question.\n"
            "Prefer making reasonable assumptions for broad exploration tasks. Do not ask for every optional slot.\n\n"
            "Priority rules:\n"
            "1. CURRENT_USER_GOAL is authoritative. Judge and normalize only this goal.\n"
            "2. BACKGROUND_RECENT_MESSAGES is only background for resolving explicit references "
            "such as `刚才`, `上一版`, `第三天`, `这个酒店`, or `加进去`.\n"
            "3. Never replace the current goal with previous task content.\n"
            "4. If the current goal mentions a new city, destination, attraction, topic, or intent "
            "that differs from recent messages, treat it as a new task.\n"
            "5. Only continue or modify a previous task when the current goal explicitly asks to "
            "continue, revise, or refer back to it.\n"
            "6. `normalized_goal` may lightly clarify pronouns or normalize wording, but must not "
            "import dense prior-plan details unless the current goal explicitly refers to them.\n"
            "7. If the current goal or a user clarification explicitly says to use the same conditions "
            "as a previous plan, expand only transferable constraints into `normalized_goal` so it is "
            "useful for search and planning.\n"
            "8. Transferable constraints include traveler count/profile, trip length, budget range, "
            "travel pace, interests, accessibility/dietary constraints, and service needs. Reuse dates "
            "only when the user clearly asks to keep the same dates.\n"
            "9. Non-transferable details include the previous destination city, previous attractions or "
            "POIs, hotels, restaurants, routes, local weather, local transport details, and day-by-day "
            "itinerary items. Never copy these into a new destination goal.\n"
            "10. When expanding inherited conditions, keep the current destination/topic authoritative "
            "and write one concise operational sentence suitable for planner and search queries.\n"
            "11. If unsure whether prior context is needed, keep `normalized_goal` close to CURRENT_USER_GOAL.\n\n"
            "Traveler profile candidate rules:\n"
            "1. Extract candidates only from explicit statements inside CURRENT_USER_GOAL, including embedded USER CLARIFICATIONS.\n"
            "2. Never infer preferences from recommendations, tools, prior assistant messages, or the normalized goal itself.\n"
            "3. Use hard_constraints only for explicit accessibility, health, allergy, dietary, or non-negotiable restrictions.\n"
            "4. Use dynamic_preferences only for clearly lasting preferences such as `I usually`, `I always`, `I like`, or `from now on`.\n"
            "5. Use trip_state for this trip, today, the current destination, temporary budget, pace, energy, or current interests.\n"
            "6. Use history_facts only for facts the user explicitly says have already happened; never store planned recommendations as history.\n"
            "7. source_text must be an exact quote copied from CURRENT_USER_GOAL. If there is no explicit evidence, return no candidate.\n"
            "8. Use operation=set to replace one field, append to add unique list values, and remove to revoke a field or listed values.\n"
            "9. Use scope=current_trip only with trip_state; all other sections use scope=long_term.\n\n"
            "Use RUNTIME_CONSTRAINTS.current_date/current_datetime/timezone as the runtime date context. "
            "Do not infer current dates from model training data.\n\n"
            f"<CURRENT_USER_GOAL>\n{context.goal}\n</CURRENT_USER_GOAL>\n\n"
            f"<BACKGROUND_RECENT_MESSAGES>\n{json.dumps(list(context.recent_messages[-6:]), ensure_ascii=False)}\n</BACKGROUND_RECENT_MESSAGES>\n\n"
            f"<RUNTIME_CONSTRAINTS>\n{json.dumps(context.constraints, ensure_ascii=False)}\n</RUNTIME_CONSTRAINTS>\n\n"
            f"<METADATA>\n{json.dumps(context.metadata, ensure_ascii=False)}\n</METADATA>\n\n"
            "Return schema:\n"
            '{"need_user_input": boolean, "question": string, "reason": string, "normalized_goal": string, '
            '"profile_candidates": [{"section": "hard_constraints|dynamic_preferences|trip_state|history_facts", '
            '"field": string, "operation": "set|append|remove", "value": any, '
            '"scope": "long_term|current_trip", "source_text": string}]}'
        )
        result = _invoke_model(model, [SystemMessage(content="Return valid JSON only."), HumanMessage(content=prompt)])
        payload = _json_payload(result)
        if not isinstance(payload, Mapping):
            msg = "Clarification judge did not return a JSON object."
            raise TypeError(msg)
        return cast("Mapping[str, object]", payload)


class _TaskGraphState(TypedDict, total=False):
    task: TaskRecord
    messages: Sequence[object]
    forced_intent: TaskIntent | None
    disabled_tools: tuple[str, ...]
    intent: TaskIntent
    context: TaskContext
    plan: TaskPlan
    queue: tuple[TaskStepRecord, ...]
    current_step: TaskStepRecord | None
    completed: tuple[TaskStepRecord, ...]
    final_answer: str
    profile_candidates: tuple[ProfileCandidate, ...]
    terminal_status: _TerminalStatus
    route: _Route
    run_state: _GraphRunState
    last_error: BaseException | None
    failed_step: TaskStepRecord | None
    ask_user_question: str
    ask_user_reason: str


class _CompiledTaskGraph(Protocol):
    def invoke(self, state: _TaskGraphState) -> object:
        """Invoke the compiled task graph."""
        ...


@dataclass(kw_only=True)
class _GraphRunState:
    started_at: float
    replan_count: int = 0
    tool_calls: int = 0
    error_counts: dict[str, int] | None = None

    def error_count(self, error: Exception) -> int:
        """Record and return the count for an error signature."""
        if self.error_counts is None:
            self.error_counts = {}
        key = _error_key(error)
        count = self.error_counts.get(key, 0) + 1
        self.error_counts[key] = count
        return count


class GraphTaskRuntime(TaskRuntime):
    """Task runtime that executes the planning loop through a LangGraph graph."""

    def __init__(
        self,
        *,
        store: TaskStore | None = None,
        router: IntentRouter | None = None,
        context_builder: ContextBuilder | None = None,
        clarification_judge: ClarificationJudge | None = None,
        planner: LLMPlanner | None = None,
        validator: PlanValidator | None = None,
        executor: TaskStepExecutor | None = None,
        observer: Observer | None = None,
        presearcher: TaskPreSearcher | None = None,
        profile_service: TravelProfileService | None = None,
        limits: TaskRuntimeLimits | None = None,
    ) -> None:
        """Initialize the graph runtime."""
        super().__init__(
            store=store,
            router=router,
            context_builder=context_builder,
            planner=planner,
            validator=validator,
            executor=executor,
            observer=observer,
            presearcher=presearcher,
            limits=limits,
        )
        self._clarification_judge = clarification_judge or ClarificationJudge()
        self._profile_service = profile_service
        self._graph = _build_task_graph(self)

    @classmethod
    def from_config(cls, config: AgentRuntimeConfig) -> GraphTaskRuntime:
        """Create the default graph runtime for deployed API servers."""

        def model_factory() -> _Model:
            from kyuriagents.runtime.dashscope import create_dashscope_model  # noqa: PLC0415

            return cast("_Model", create_dashscope_model(config))

        store: TaskStore = PostgresTaskStore(dsn=config.postgres_dsn) if config.postgres_dsn else InMemoryTaskStore()
        limits = TaskRuntimeLimits()
        tool_executor = TaskToolExecutor.from_config(config)
        evidence_agents = create_evidence_agents(config)
        profile_service = None
        if config.enable_travel_profile and config.postgres_dsn:
            from kyuriagents.profile import PostgresTravelProfileStore, TravelProfileService  # noqa: PLC0415

            profile_service = TravelProfileService(PostgresTravelProfileStore(dsn=config.postgres_dsn))
        return cls(
            store=store,
            planner=LLMPlanner(model_factory=model_factory),
            clarification_judge=ClarificationJudge(model_factory=model_factory),
            executor=TaskStepExecutor(model_factory=model_factory, tool_executor=tool_executor, evidence_agents=evidence_agents),
            context_builder=ContextBuilder(profile_service=profile_service),
            presearcher=TaskPreSearcher.from_config(config),
            profile_service=profile_service,
            limits=limits,
        )

    def run_existing_task(
        self,
        *,
        task: TaskRecord,
        messages: Sequence[MessageRecord] = (),
        forced_intent: TaskIntent | None = "task",
        disabled_tools: Sequence[str] = (),
    ) -> TaskRunResult:
        """Run a persisted task through the LangGraph planning loop."""
        state: _TaskGraphState = {
            "task": task,
            "messages": messages,
            "forced_intent": forced_intent,
            "disabled_tools": tuple(disabled_tools),
            "completed": (),
            "final_answer": "",
            "profile_candidates": (),
            "terminal_status": "pending",
            "route": "plan",
            "run_state": _GraphRunState(started_at=time.monotonic()),
            "last_error": None,
            "failed_step": None,
        }
        try:
            final_state = cast("_TaskGraphState", self._graph.invoke(state))
            task = final_state.get("task", task)
        except Exception as exc:  # noqa: BLE001  # task state must record unexpected graph errors
            task = self.store.update_task(task.task_id, status="failed", error_message=str(exc), finished=True)
            self.store.add_event(task_id=task.task_id, event_type="failed", message=str(exc))
        return self._result(task)

    def _build_context_node(self, state: _TaskGraphState) -> _TaskGraphState:
        task = state["task"]
        intent = self._router.route(task.goal, forced_intent=state.get("forced_intent"))
        task = self.store.update_task(task.task_id, status="planning", intent=intent)
        self.store.add_event(task_id=task.task_id, event_type="intent", message=f"Intent routed as {intent}.")
        context = self._context_builder.build(
            goal=task.goal,
            intent=intent,
            tenant_id=task.tenant_id,
            user_id=task.user_id,
            thread_id=task.thread_id,
            messages=cast("Sequence[MessageRecord]", state.get("messages", ())),
            tool_descriptors=_filter_descriptors(self._executor.tool_descriptors, disabled_tools=state.get("disabled_tools", ())),
            constraints={"max_steps": self._limits.max_plan_steps, "max_tool_calls": self._limits.max_tool_calls},
        )
        self.store.add_event(task_id=task.task_id, event_type="context", message="Task context built.")
        decision = self._clarification_judge.judge(context)
        if decision.need_user_input:
            question = decision.question or "I need one more detail before I can plan this task."
            metadata = _hitl_requested_metadata(task, question=question, reason=decision.reason)
            task = self.store.update_task(
                task.task_id,
                status="waiting_user",
                final_answer=question,
                metadata=metadata,
            )
            self.store.add_event(
                task_id=task.task_id,
                event_type="hitl_requested",
                message="Task is waiting for user clarification.",
                payload={"question": question, "reason": decision.reason},
            )
            return {**state, "task": task, "intent": intent, "context": context, "route": "ask_user", "ask_user_question": question}
        profile_candidates = _user_sourced_profile_candidates(decision.profile_candidates, goal=context.goal)
        if decision.normalized_goal and decision.normalized_goal != context.goal:
            context = TaskContext(
                goal=decision.normalized_goal,
                intent=context.intent,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                thread_id=context.thread_id,
                recent_messages=context.recent_messages,
                available_tools=context.available_tools,
                constraints=context.constraints,
                metadata={**context.metadata, "original_goal": task.goal},
            )
        return {
            **state,
            "task": task,
            "intent": intent,
            "context": context,
            "profile_candidates": profile_candidates,
            "route": "presearch",
        }

    def _presearch_node(self, state: _TaskGraphState) -> _TaskGraphState:
        task = state["task"]
        context = self._presearcher.enrich(state["context"])
        self._record_presearch_event(task, context)
        return {**state, "context": context, "route": "plan"}

    def _planner_node(self, state: _TaskGraphState) -> _TaskGraphState:
        task = state["task"]
        context = state["context"]
        plan = _normalize_plan(self._planner.plan(context), limits=self._limits)
        validation = self._validator.validate(plan, context)
        if not validation.valid:
            message = "; ".join(validation.errors)
            task = self.store.update_task(task.task_id, status="failed", error_message=message, finished=True)
            self.store.add_event(task_id=task.task_id, event_type="failed", message=message)
            return {
                **state,
                "task": task,
                "plan": plan,
                "queue": (),
                "route": "fail",
                "final_answer": message,
                "terminal_status": "failed",
            }
        steps = self.store.add_steps(task_id=task.task_id, steps=_records_from_plan(task.task_id, plan))
        self.store.add_event(task_id=task.task_id, event_type="planned", message=plan.summary or "Plan generated.")
        self.store.add_event(task_id=task.task_id, event_type="validated", message="Plan validated.")
        task = self.store.update_task(task.task_id, status="running")
        return {**state, "task": task, "plan": plan, "queue": tuple(steps), "route": "select_step"}

    def _select_step_node(self, state: _TaskGraphState) -> _TaskGraphState:
        task = state["task"]
        run_state = state["run_state"]
        self._check_graph_runtime_budget(run_state)
        queue = tuple(self.store.list_steps(task_id=task.task_id))
        completed = tuple(step for step in queue if step.status in {"succeeded", "skipped"})
        pending = next((step for step in queue if step.status == "pending"), None)
        if pending is None:
            terminal_status: _TerminalStatus = (
                "succeeded" if any(step.kind == "answer" and step.status == "succeeded" for step in completed) else "failed"
            )
            return {
                **state,
                "queue": queue,
                "completed": completed,
                "current_step": None,
                "terminal_status": terminal_status,
                "route": "answer",
            }
        if _counts_toward_tool_limit(pending) and run_state.tool_calls >= self._limits.max_tool_calls:
            skipped = self._skip_step_for_tool_limit(task=task, step=pending)
            completed = (*completed, skipped)
            return {**state, "queue": tuple(self.store.list_steps(task_id=task.task_id)), "completed": completed, "route": "select_step"}
        return {**state, "queue": queue, "completed": completed, "current_step": pending, "route": "execute_step", "last_error": None}

    def _execute_step_node(self, state: _TaskGraphState) -> _TaskGraphState:
        task = state["task"]
        current = state.get("current_step")
        if current is None:
            return {**state, "terminal_status": "failed", "route": "answer"}
        run_state = state["run_state"]
        context = state["context"]
        execution_context = TaskExecutionContext(
            tenant_id=task.tenant_id,
            user_id=task.user_id,
            thread_id=task.thread_id,
            goal=context.goal,
            defaults=RuntimeContextDefaults(tenant_id=task.tenant_id, user_id=task.user_id, thread_id=task.thread_id),
        )
        running = self.store.update_step(current.step_id, status="running", attempts=current.attempts + 1, started=True)
        self.store.add_event(task_id=task.task_id, step_id=running.step_id, event_type="step_started", message=running.title)
        try:
            self._check_graph_runtime_budget(run_state)
            if _counts_toward_tool_limit(running):
                if run_state.tool_calls >= self._limits.max_tool_calls:
                    output = f"Task reached tool-call limit of {self._limits.max_tool_calls}."
                else:
                    run_state.tool_calls += 1
                    output = self._executor.execute(running, context=execution_context, previous_steps=state.get("completed", ()))
            else:
                output = self._executor.execute(running, context=execution_context, previous_steps=state.get("completed", ()))
        except Exception as exc:  # noqa: BLE001  # the observer node turns failures into routes
            return {**state, "current_step": running, "last_error": exc, "route": "execute_step"}
        output = _truncate_text(output, max_chars=self._limits.max_step_output_chars)
        finished = self.store.update_step(running.step_id, status="succeeded", output=output, error_message=None, finished=True)
        self.store.add_event(task_id=task.task_id, step_id=finished.step_id, event_type="step_finished", message=finished.title)
        completed = (*state.get("completed", ()), finished)
        final_answer = output if finished.kind == "answer" else state.get("final_answer", "")
        terminal_status = "succeeded" if finished.kind == "answer" else state.get("terminal_status", "pending")
        return {
            **state,
            "current_step": finished,
            "completed": completed,
            "final_answer": final_answer,
            "terminal_status": terminal_status,
            "last_error": None,
        }

    def _observer_node(self, state: _TaskGraphState) -> _TaskGraphState:
        task = state["task"]
        current = state.get("current_step")
        if current is None:
            return {**state, "route": "answer"}
        error = state.get("last_error")
        if error is None:
            observation = self._observer.observe(current)
            route: _Route = "answer" if observation.decision == "finish" else "select_step"
            return {**state, "route": route}
        if not isinstance(error, Exception):
            error = Exception(str(error))
        run_state = state["run_state"]
        error_count = run_state.error_count(error)
        queue = state.get("queue", ())
        index = _step_index(queue, current.step_id)
        observation = self._observer.observe(
            current,
            error=error,
            limits=self._limits,
            can_replan=self._can_replan_graph(run_state, current=current, queue=queue, index=index),
            same_error_exceeded=error_count > self._limits.max_same_error,
        )
        if observation.decision == "retry":
            self.store.add_event(task_id=task.task_id, step_id=current.step_id, event_type="retry", message=observation.message)
            return {**state, "route": "execute_step"}
        if observation.decision == "replan":
            failed = self.store.update_step(current.step_id, status="failed", error_message=str(error), finished=True)
            self.store.add_event(task_id=task.task_id, step_id=failed.step_id, event_type="failed", message=str(error))
            return {**state, "failed_step": failed, "current_step": failed, "route": "replan"}
        if observation.decision == "skip" or _can_skip_failed_tool(current):
            skipped = self._skip_failed_step(task=task, step=current, message=observation.message, error=error)
            return {**state, "completed": (*state.get("completed", ()), skipped), "current_step": skipped, "route": "select_step"}
        message = f"Task step failed: {error}"
        failed = self.store.update_step(current.step_id, status="failed", error_message=str(error), finished=True)
        self.store.add_event(task_id=task.task_id, step_id=failed.step_id, event_type="failed", message=message)
        return {
            **state,
            "current_step": failed,
            "failed_step": failed,
            "final_answer": message,
            "terminal_status": "failed",
            "route": "answer",
        }

    def _replanner_node(self, state: _TaskGraphState) -> _TaskGraphState:
        task = state["task"]
        context = state["context"]
        run_state = state["run_state"]
        failed_step = state.get("failed_step") or state.get("current_step")
        error = state.get("last_error")
        if failed_step is None or not isinstance(error, Exception):
            return {**state, "terminal_status": "failed", "route": "answer"}
        queue = state.get("queue", ())
        index = _step_index(queue, failed_step.step_id)
        self._skip_remaining(task=task, steps=queue[index + 1 :])
        replanned = self._replan_graph(
            run_state=run_state,
            task=task,
            context=context,
            failed_step=failed_step,
            completed=state.get("completed", ()),
            error=error,
        )
        if not replanned:
            return {
                **state,
                "route": "answer",
                "final_answer": f"Task failed while replanning: {error}",
                "terminal_status": "failed",
            }
        return {
            **state,
            "queue": tuple(self.store.list_steps(task_id=task.task_id)),
            "current_step": None,
            "failed_step": None,
            "last_error": None,
            "route": "select_step",
        }

    def _answer_node(self, state: _TaskGraphState) -> _TaskGraphState:
        task = state["task"]
        if task.status == "waiting_user":
            return state
        final_answer = (
            state.get("final_answer") or _previous_step_text(state.get("completed", ())) or "Task completed, but no usable result was generated."
        )
        status = "succeeded" if state.get("terminal_status") == "succeeded" else "failed"
        if status == "succeeded":
            self._commit_profile_candidates(state)
        task = self.store.update_task(task.task_id, status=status, final_answer=final_answer, finished=True)
        self.store.add_event(task_id=task.task_id, event_type="finished", message="Task completed." if status == "succeeded" else "Task failed.")
        return {**state, "task": task, "final_answer": final_answer}

    def _commit_profile_candidates(self, state: _TaskGraphState) -> None:
        candidates = state.get("profile_candidates", ())
        if self._profile_service is None or not candidates:
            return
        task = state["task"]
        try:
            record = self._profile_service.apply_candidates(
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                candidates=candidates,
            )
        except Exception as exc:  # noqa: BLE001  # profile persistence must not invalidate a completed task
            self.store.add_event(
                task_id=task.task_id,
                event_type="profile_update_failed",
                message=f"Traveler profile update failed: {exc}",
            )
            return
        if record is None:
            return
        self.store.add_event(
            task_id=task.task_id,
            event_type="profile_updated",
            message="Traveler profile updated from explicit task input.",
            payload={
                "profile_version": record.profile_version,
                "candidate_count": len(candidates),
                "changes": [_profile_candidate_payload(candidate) for candidate in candidates],
            },
        )

    def _check_graph_runtime_budget(self, run_state: _GraphRunState) -> None:
        elapsed = time.monotonic() - run_state.started_at
        if elapsed > self._limits.max_runtime_seconds:
            msg = f"Task exceeded runtime limit of {self._limits.max_runtime_seconds:.1f}s."
            raise TimeoutError(msg)

    def _can_replan_graph(self, run_state: _GraphRunState, *, current: TaskStepRecord, queue: Sequence[TaskStepRecord], index: int) -> bool:
        if current.kind != "tool" or not current.tool_name:
            return False
        if run_state.replan_count >= self._limits.max_replans:
            return False
        existing_steps = len(self.store.list_steps(task_id=current.task_id))
        if existing_steps >= self._limits.max_total_steps:
            return False
        return any(step.status == "pending" for step in queue[index + 1 :])

    def _replan_graph(
        self,
        *,
        run_state: _GraphRunState,
        task: TaskRecord,
        context: TaskContext,
        failed_step: TaskStepRecord,
        completed: Sequence[TaskStepRecord],
        error: Exception,
    ) -> list[TaskStepRecord]:
        existing_steps = self.store.list_steps(task_id=task.task_id)
        remaining_budget = self._limits.max_total_steps - len(existing_steps)
        if remaining_budget <= 0:
            return []
        run_state.replan_count += 1
        replan_context = _replan_context(context, failed_step=failed_step, completed=completed, error=error)
        plan = _normalize_plan(self._planner.plan(replan_context), limits=self._limits, max_steps=remaining_budget)
        validation = self._validator.validate(plan, replan_context)
        if not validation.valid:
            message = "; ".join(validation.errors)
            self.store.add_event(task_id=task.task_id, event_type="failed", message=f"Replan rejected: {message}")
            return []
        start_index = len(existing_steps)
        records = self.store.add_steps(task_id=task.task_id, steps=_records_from_plan(task.task_id, plan, start_index=start_index))
        self.store.add_event(
            task_id=task.task_id,
            step_id=failed_step.step_id,
            event_type="replanned",
            message=plan.summary or f"Replanned after `{failed_step.title}` failed.",
            payload={"failed_step_id": failed_step.step_id, "failed_tool": failed_step.tool_name, "error": str(error)},
        )
        return records


def _build_task_graph(runtime: GraphTaskRuntime) -> _CompiledTaskGraph:
    graph = StateGraph(_TaskGraphState)  # ty: ignore[invalid-argument-type]  # ty cannot verify TypedDicts satisfy LangGraph's StateLike bound
    graph.add_node("context", runtime._build_context_node)
    graph.add_node("presearch", runtime._presearch_node)
    graph.add_node("planner", runtime._planner_node)
    graph.add_node("select_step", runtime._select_step_node)
    graph.add_node("execute_step", runtime._execute_step_node)
    graph.add_node("observer", runtime._observer_node)
    graph.add_node("replanner", runtime._replanner_node)
    graph.add_node("answer", runtime._answer_node)
    graph.add_edge(START, "context")
    graph.add_conditional_edges("context", _route, {"ask_user": END, "presearch": "presearch"})
    graph.add_conditional_edges("presearch", _route, {"plan": "planner"})
    graph.add_conditional_edges("planner", _route, {"select_step": "select_step", "fail": "answer"})
    graph.add_conditional_edges("select_step", _route, {"select_step": "select_step", "execute_step": "execute_step", "answer": "answer"})
    graph.add_edge("execute_step", "observer")
    graph.add_conditional_edges(
        "observer",
        _route,
        {
            "execute_step": "execute_step",
            "select_step": "select_step",
            "replan": "replanner",
            "ask_user": END,
            "answer": "answer",
        },
    )
    graph.add_conditional_edges("replanner", _route, {"select_step": "select_step", "answer": "answer"})
    graph.add_edge("answer", END)
    return cast("_CompiledTaskGraph", graph.compile())


def _route(state: _TaskGraphState) -> _Route:
    return state.get("route", "answer")


def _step_index(queue: Sequence[TaskStepRecord], step_id: str) -> int:
    for index, step in enumerate(queue):
        if step.step_id == step_id:
            return index
    return max(0, len(queue) - 1)


def _context_for_clarification(context: TaskContext) -> dict[str, object]:
    return {
        "goal": context.goal,
        "intent": context.intent,
        "recent_messages": context.recent_messages[-6:],
        "constraints": context.constraints,
        "metadata": context.metadata,
    }


def _hitl_requested_metadata(task: TaskRecord, *, question: str, reason: str) -> dict[str, object]:
    metadata = dict(task.metadata)
    original_goal = str(metadata.get("original_goal") or task.goal)
    existing = metadata.get("hitl")
    hitl = dict(existing) if isinstance(existing, Mapping) else {}
    answers = hitl.get("answers")
    metadata["original_goal"] = original_goal
    metadata["clarification_reason"] = reason
    metadata["hitl"] = {
        **hitl,
        "status": "waiting_user",
        "source": "clarification",
        "question": question,
        "reason": reason,
        "answers": list(answers) if isinstance(answers, list) else [],
    }
    return metadata


def _clarification_from_payload(payload: Mapping[str, object], *, fallback_goal: str) -> ClarificationDecision:
    return ClarificationDecision(
        need_user_input=bool(payload.get("need_user_input")),
        question=str(payload.get("question") or "").strip(),
        reason=str(payload.get("reason") or "").strip(),
        normalized_goal=str(payload.get("normalized_goal") or fallback_goal).strip() or fallback_goal,
        profile_candidates=_profile_candidates_from_payload(payload.get("profile_candidates")),
    )


def _profile_candidates_from_payload(value: object) -> tuple[ProfileCandidate, ...]:
    if not isinstance(value, list):
        return ()
    candidates: list[ProfileCandidate] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            candidate = normalize_profile_candidate(item)
        except ValueError:
            continue
        candidates.append(candidate)
    return tuple(candidates)


def _user_sourced_profile_candidates(candidates: Sequence[ProfileCandidate], *, goal: str) -> tuple[ProfileCandidate, ...]:
    normalized_goal = _normalized_source_text(goal)
    accepted: list[ProfileCandidate] = []
    for candidate in candidates:
        source = _normalized_source_text(candidate.source_text)
        if source and source in normalized_goal and candidate not in accepted:
            accepted.append(candidate)
    return tuple(accepted)


def _normalized_source_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _profile_candidate_payload(candidate: ProfileCandidate) -> dict[str, object]:
    return {
        "section": candidate.section,
        "field": candidate.field,
        "operation": candidate.operation,
        "value": candidate.value,
        "scope": candidate.scope,
        "source_text": candidate.source_text,
    }


__all__ = ["ClarificationDecision", "ClarificationJudge", "GraphTaskRuntime"]
