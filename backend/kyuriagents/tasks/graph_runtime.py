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
from kyuriagents.tasks.evidence import create_evidence_agents
from kyuriagents.tasks.runtime import (
    ContextBuilder,
    IntentRouter,
    LLMPlanner,
    Observer,
    PlanValidator,
    TaskExecutionContext,
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

_Route = Literal["plan", "select_step", "execute_step", "replan", "ask_user", "answer", "fail"]


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
    """

    need_user_input: bool = False
    question: str = ""
    reason: str = ""
    normalized_goal: str = ""


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
            "Ask the user only when the task cannot be usefully planned without one short follow-up question.\n"
            "Prefer making reasonable assumptions for broad exploration tasks. Do not ask for every optional slot.\n\n"
            f"Context:\n{json.dumps(_context_for_clarification(context), ensure_ascii=False)}\n\n"
            "Return schema:\n"
            '{"need_user_input": boolean, "question": string, "reason": string, "normalized_goal": string}'
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
            limits=limits,
        )
        self._clarification_judge = clarification_judge or ClarificationJudge()
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
        return cls(
            store=store,
            planner=LLMPlanner(model_factory=model_factory),
            clarification_judge=ClarificationJudge(model_factory=model_factory),
            executor=TaskStepExecutor(model_factory=model_factory, tool_executor=tool_executor, evidence_agents=evidence_agents),
            context_builder=ContextBuilder(),
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
        return {**state, "task": task, "intent": intent, "context": context, "route": "plan"}

    def _planner_node(self, state: _TaskGraphState) -> _TaskGraphState:
        task = state["task"]
        context = state["context"]
        plan = _normalize_plan(self._planner.plan(context), limits=self._limits)
        steps = self.store.add_steps(task_id=task.task_id, steps=_records_from_plan(task.task_id, plan))
        self.store.add_event(task_id=task.task_id, event_type="planned", message=plan.summary or "Plan generated.")
        validation = self._validator.validate(plan, context)
        if not validation.valid:
            message = "; ".join(validation.errors)
            task = self.store.update_task(task.task_id, status="failed", error_message=message, finished=True)
            self.store.add_event(task_id=task.task_id, event_type="failed", message=message)
            return {**state, "task": task, "plan": plan, "queue": tuple(steps), "route": "fail", "final_answer": message}
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
            return {**state, "queue": queue, "completed": completed, "current_step": None, "route": "answer"}
        if _counts_toward_tool_limit(pending) and run_state.tool_calls >= self._limits.max_tool_calls:
            skipped = self._skip_step_for_tool_limit(task=task, step=pending)
            completed = (*completed, skipped)
            return {**state, "queue": tuple(self.store.list_steps(task_id=task.task_id)), "completed": completed, "route": "select_step"}
        return {**state, "queue": queue, "completed": completed, "current_step": pending, "route": "execute_step", "last_error": None}

    def _execute_step_node(self, state: _TaskGraphState) -> _TaskGraphState:
        task = state["task"]
        current = state.get("current_step")
        if current is None:
            return {**state, "route": "answer"}
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
        return {**state, "current_step": finished, "completed": completed, "final_answer": final_answer, "last_error": None}

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
        return {**state, "current_step": failed, "failed_step": failed, "final_answer": message, "route": "answer"}

    def _replanner_node(self, state: _TaskGraphState) -> _TaskGraphState:
        task = state["task"]
        context = state["context"]
        run_state = state["run_state"]
        failed_step = state.get("failed_step") or state.get("current_step")
        error = state.get("last_error")
        if failed_step is None or not isinstance(error, Exception):
            return {**state, "route": "answer"}
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
        status = "failed" if state.get("route") == "fail" else "succeeded"
        task = self.store.update_task(task.task_id, status=status, final_answer=final_answer, finished=True)
        self.store.add_event(task_id=task.task_id, event_type="finished", message="Task completed." if status == "succeeded" else "Task failed.")
        return {**state, "task": task, "final_answer": final_answer}

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
    graph.add_node("planner", runtime._planner_node)
    graph.add_node("select_step", runtime._select_step_node)
    graph.add_node("execute_step", runtime._execute_step_node)
    graph.add_node("observer", runtime._observer_node)
    graph.add_node("replanner", runtime._replanner_node)
    graph.add_node("answer", runtime._answer_node)
    graph.add_edge(START, "context")
    graph.add_conditional_edges("context", _route, {"ask_user": END, "plan": "planner"})
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
    )


__all__ = ["ClarificationDecision", "ClarificationJudge", "GraphTaskRuntime"]
