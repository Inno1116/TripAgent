"""Task-mode traveler profile candidate integration tests."""

from __future__ import annotations

from dataclasses import replace

from kyuriagents.profile import ProfileCandidate, TravelProfileRecord, TravelProfileService, default_profile_data
from kyuriagents.tasks import (
    ClarificationDecision,
    ContextBuilder,
    GraphTaskRuntime,
    InMemoryTaskStore,
    PlannedStep,
    TaskPlan,
    TaskPreSearcher,
    TaskRuntimeLimits,
    TaskStepExecutor,
    TaskToolExecutor,
)


class _ProfileStore:
    def __init__(self) -> None:
        self.record: TravelProfileRecord | None = None

    def get(self, *, tenant_id: str, user_id: str) -> TravelProfileRecord | None:
        _ = tenant_id, user_id
        return self.record

    def upsert(
        self,
        *,
        tenant_id: str,
        user_id: str,
        profile_data: dict[str, object],
        expected_version: int | None = None,
    ) -> TravelProfileRecord:
        if self.record is not None and expected_version is not None:
            assert self.record.profile_version == expected_version
        version = 1 if self.record is None else self.record.profile_version + 1
        self.record = TravelProfileRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            profile_data=profile_data,
            profile_version=version,
        )
        return self.record


class _ClarificationJudge:
    def judge(self, context: object) -> ClarificationDecision:
        goal = str(getattr(context, "goal"))
        return ClarificationDecision(
            normalized_goal=goal,
            profile_candidates=(
                ProfileCandidate(
                    section="trip_state",
                    field="current_interests",
                    operation="set",
                    value=["名胜古迹"],
                    scope="current_trip",
                    source_text="这次北京想看名胜古迹",
                ),
            ),
        )


class _Planner:
    def plan(self, context: object) -> TaskPlan:
        return TaskPlan(
            goal=str(getattr(context, "goal")),
            steps=(
                PlannedStep(
                    kind="answer",
                    title="生成最终答案",
                    instruction="根据用户目标生成最终答案。",
                ),
            ),
        )


class _AnswerModel:
    def invoke(self, input_data: object) -> str:
        _ = input_data
        return "北京名胜古迹行程已经生成。"


class _FailingModel:
    def invoke(self, input_data: object) -> str:
        _ = input_data
        msg = "answer generation failed"
        raise RuntimeError(msg)


def test_successful_task_commits_explicit_profile_candidate() -> None:
    profile_store = _ProfileStore()
    profile_store.record = TravelProfileRecord(
        tenant_id="default",
        user_id="user-1",
        profile_data={
            **default_profile_data(),
            "dynamic_preferences": {"food": ["本地美食"]},
        },
        profile_version=1,
    )
    result = _run_task(profile_store=profile_store, model=_AnswerModel())

    assert result.task.status == "succeeded"
    assert profile_store.record is not None
    assert profile_store.record.profile_version == 2
    assert profile_store.record.profile_data["dynamic_preferences"] == {"food": ["本地美食"]}
    assert profile_store.record.profile_data["trip_state"] == {"current_interests": ["名胜古迹"]}
    assert any(event.event_type == "profile_updated" for event in result.events)


def test_failed_task_does_not_commit_profile_candidate() -> None:
    profile_store = _ProfileStore()
    result = _run_task(profile_store=profile_store, model=_FailingModel())

    assert result.task.status == "failed"
    assert profile_store.record is None
    assert not any(event.event_type == "profile_updated" for event in result.events)


def _run_task(*, profile_store: _ProfileStore, model: object) -> object:
    store = InMemoryTaskStore()
    profile_service = TravelProfileService(profile_store)
    runtime = GraphTaskRuntime(
        store=store,
        context_builder=ContextBuilder(profile_service=profile_service),
        clarification_judge=_ClarificationJudge(),
        planner=_Planner(),
        executor=TaskStepExecutor(
            model_factory=lambda: model,
            tool_executor=TaskToolExecutor(handlers={}, descriptors=()),
        ),
        presearcher=TaskPreSearcher(),
        profile_service=profile_service,
        limits=TaskRuntimeLimits(max_step_retries=0),
    )
    task = store.create_task(
        tenant_id="default",
        user_id="user-1",
        thread_id="thread-1",
        goal="这次北京想看名胜古迹",
    )
    return runtime.run_existing_task(task=replace(task))
