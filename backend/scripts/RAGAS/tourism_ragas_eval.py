"""Run end-to-end RAGAS / LLM-judge evaluation for the tourism RAG set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections.abc import Mapping, Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal, cast

from eval_common import (
    aggregate_metric_dicts,
    default_embedding_file,
    default_eval_file,
    infer_scope_from_kb,
    load_eval_cases,
    load_query_embeddings,
    load_runtime_env,
    normalize_query,
    write_jsonl,
)
from kyuriagents.rag import ElasticsearchKeywordStore, MilvusVectorStore, PostgresChunkTextHydrator, RetrievalScope
from kyuriagents.rag.types import RetrievedChunk
from kyuriagents.runtime import AgentRuntimeConfig
from kyuriagents.runtime.dashscope import create_dashscope_model
from tourism_retrieval_eval import CachedQueryEmbedder, Mode, _retriever_for_mode, _retrieve

_DEFAULT_GENERATIONS_FILE = Path(__file__).resolve().parent / "tourism_ragas_generations.jsonl"
_DEFAULT_RAGAS_SCORES_FILE = Path(__file__).resolve().parent / "tourism_ragas_scores.jsonl"
_DEFAULT_JUDGE_SCORES_FILE = Path(__file__).resolve().parent / "tourism_llm_judge_scores.jsonl"
_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise Chinese tourism QA assistant. Answer only from the provided contexts. "
    "If the contexts are insufficient, say so briefly. Keep the answer concise and factual."
)
JudgeMode = Literal["ragas", "llm"]


def main() -> None:
    """Run generation and judge evaluation."""
    parser = argparse.ArgumentParser(description="Generate RAG answers and evaluate them with RAGAS or a compact LLM judge.")
    parser.add_argument("--eval-file", type=Path, default=default_eval_file(), help="Annotated tourism eval JSONL.")
    parser.add_argument("--embeddings-file", type=Path, default=default_embedding_file(), help="Cached query embedding JSONL.")
    parser.add_argument("--generations-file", type=Path, default=_DEFAULT_GENERATIONS_FILE, help="Cached generated answers JSONL.")
    parser.add_argument("--scores-file", type=Path, default=None, help="Output score JSONL. Defaults depend on --judge.")
    parser.add_argument("--kb-id", default="", help="Knowledge base id. Defaults to RAG_KB_IDS when set.")
    parser.add_argument("--tenant-id", default="", help="Tenant id override.")
    parser.add_argument("--user-id", default="", help="User id override; needed for private uploaded documents.")
    parser.add_argument("--mode", choices=("vector", "hybrid", "hybrid_rerank"), default="hybrid_rerank", help="Retrieval mode used for contexts.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of contexts supplied to answer generation.")
    parser.add_argument("--candidate-k", type=int, default=40, help="Vector/keyword candidates before fusion.")
    parser.add_argument("--rerank-candidates", type=int, default=40, help="Hybrid candidates sent to reranker.")
    parser.add_argument("--answer-model", default="", help="Model used to generate answers. Defaults to RAGAS_ANSWER_MODEL or DASHSCOPE_CHAT_MODEL.")
    parser.add_argument("--judge-model", default="", help="Model used by judge/RAGAS. Defaults to RAGAS_JUDGE_MODEL or DASHSCOPE_CHAT_MODEL.")
    parser.add_argument("--phase", choices=("generate", "judge", "all"), default="all", help="Run answer generation, judging, or both.")
    parser.add_argument("--judge", choices=("ragas", "llm"), default="ragas", help="Use official RAGAS metrics or a compact JSON LLM judge.")
    parser.add_argument(
        "--ragas-metrics",
        default="faithfulness,answer_correctness,context_recall",
        help=(
            "Comma-separated RAGAS metrics. Supported: faithfulness,answer_relevancy,"
            "context_precision,context_recall,answer_correctness."
        ),
    )
    parser.add_argument("--ragas-max-workers", type=int, default=1, help="RAGAS concurrent workers. Use 1 for DashScope free-tier stability.")
    parser.add_argument("--ragas-max-retries", type=int, default=1, help="RAGAS retry attempts per LLM operation.")
    parser.add_argument("--ragas-timeout", type=int, default=180, help="RAGAS timeout seconds per operation.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of cases for a smoke test.")
    parser.add_argument("--force", action="store_true", help="Regenerate answers even if they already exist.")
    args = parser.parse_args()

    if args.top_k <= 0:
        msg = "--top-k must be positive"
        raise ValueError(msg)

    load_runtime_env()
    config = AgentRuntimeConfig.from_env()
    answer_model = args.answer_model or os.environ.get("RAGAS_ANSWER_MODEL") or None
    judge_model = args.judge_model or os.environ.get("RAGAS_JUDGE_MODEL") or None
    cases = load_eval_cases(args.eval_file)
    if args.limit > 0:
        cases = cases[: args.limit]
    scores_file = args.scores_file or (_DEFAULT_RAGAS_SCORES_FILE if args.judge == "ragas" else _DEFAULT_JUDGE_SCORES_FILE)

    if args.phase in ("generate", "all"):
        print(
            f"[generate] start cases={len(cases)} mode={args.mode} top_k={args.top_k} "
            f"force={args.force} answer_model={answer_model or config.chat_model}",
            flush=True,
        )
        rows = _generate_answers(args=args, config=config, cases=cases, answer_model=answer_model)
        print(
            json.dumps(
                {
                    "phase": "generate",
                    "count": len(rows),
                    "output": str(args.generations_file),
                    "mode": args.mode,
                    "top_k": args.top_k,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.phase in ("judge", "all"):
        generated_rows = _load_jsonl(args.generations_file)
        if args.limit > 0:
            generated_rows = generated_rows[: args.limit]
        print(
            f"[judge] start judge={args.judge} cases={len(generated_rows)} "
            f"judge_model={judge_model or config.chat_model} input={args.generations_file}",
            flush=True,
        )
        if args.judge == "ragas":
            score_rows, summary = _run_ragas(
                generated_rows,
                config=config,
                judge_model=judge_model,
                metric_names=_parse_ragas_metric_names(args.ragas_metrics),
                max_workers=args.ragas_max_workers,
                max_retries=args.ragas_max_retries,
                timeout=args.ragas_timeout,
            )
        else:
            score_rows, summary = _run_llm_judge(generated_rows, config=config, judge_model=judge_model)
        write_jsonl(scores_file, score_rows)
        print(json.dumps({"phase": "judge", "judge": args.judge, "count": len(score_rows), "summary": summary}, ensure_ascii=False, indent=2))
        print(f"wrote {scores_file}")


def _generate_answers(
    *,
    args: argparse.Namespace,
    config: AgentRuntimeConfig,
    cases: Sequence[object],
    answer_model: str | None,
) -> list[dict[str, object]]:
    existing = {} if args.force else {str(row.get("id")): row for row in _load_jsonl(args.generations_file) if row.get("id")}
    embedding_records = load_query_embeddings(args.embeddings_file)
    cached_embeddings = {record.normalized_query: record.embedding for record in embedding_records.values()}
    embed_query = CachedQueryEmbedder(cached_embeddings)
    kb_id = args.kb_id or (config.rag_kb_ids[0] if config.rag_kb_ids else "")
    if not kb_id:
        msg = "Set --kb-id or RAG_KB_IDS before running RAGAS evaluation."
        raise ValueError(msg)
    tenant_id, user_id = _resolve_scope(config=config, kb_id=kb_id, tenant_id=args.tenant_id, user_id=args.user_id)
    scope = RetrievalScope(tenant_id=tenant_id, user_id=user_id, kb_ids=(kb_id,))
    hydrator = PostgresChunkTextHydrator(dsn=config.postgres_dsn) if config.postgres_dsn else None
    vector_store = MilvusVectorStore(
        collection_name=config.rag_milvus_collection,
        uri=config.rag_milvus_uri,
        token=config.rag_milvus_token,
        db_name=config.rag_milvus_db,
        embed_query=embed_query,
    )
    keyword_store = ElasticsearchKeywordStore(index=config.rag_es_index, url=config.rag_es_url)
    mode = cast("Mode", args.mode)
    retriever = _retriever_for_mode(
        mode,
        config=config,
        vector_store=vector_store,
        keyword_store=keyword_store,
        hydrator=hydrator,
        candidate_k=args.candidate_k,
        rerank_candidates=args.rerank_candidates,
        top_k=args.top_k,
    )
    model = _create_role_chat_model(config, role="answer", model_name=answer_model)

    rows: list[dict[str, object]] = []
    generated = 0
    reused = 0
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        case_id = str(getattr(case, "id"))
        if case_id in existing:
            rows.append(dict(existing[case_id]))
            reused += 1
            print(f"[generate] {index}/{total} {case_id} reused", flush=True)
            continue
        question = str(getattr(case, "question"))
        print(f"[generate] {index}/{total} {case_id} retrieving...", flush=True)
        retrieved = _retrieve(mode, retriever=retriever, vector_store=vector_store, hydrator=hydrator, case_query=question, scope=scope, top_k=args.top_k)
        contexts = [_context_text(chunk) for chunk in retrieved if _context_text(chunk)]
        print(f"[generate] {index}/{total} {case_id} contexts={len(contexts)} answering...", flush=True)
        answer = _invoke_answer_model(model, question=question, contexts=contexts)
        rows.append(
            {
                "id": case_id,
                "type": str(getattr(case, "type")),
                "difficulty": str(getattr(case, "difficulty")),
                "question": question,
                "answer": answer,
                "ground_truth_answer": str(getattr(case, "ground_truth_answer")),
                "contexts": contexts,
                "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved],
                "retrieved": [_retrieved_context_row(chunk, rank=index) for index, chunk in enumerate(retrieved, start=1)],
                "mode": args.mode,
                "top_k": args.top_k,
            }
        )
        generated += 1
        print(f"[generate] {index}/{total} {case_id} done answer_chars={len(answer)}", flush=True)

    write_jsonl(args.generations_file, rows)
    print(json.dumps({"generated": generated, "reused": reused}, ensure_ascii=False))
    return rows


def _run_ragas(
    rows: Sequence[Mapping[str, object]],
    *,
    config: AgentRuntimeConfig,
    judge_model: str | None,
    metric_names: Sequence[str],
    max_workers: int,
    max_retries: int,
    timeout: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    print("[judge/ragas] loading ragas dependencies...", flush=True)
    _patch_ragas_optional_vertexai()
    try:
        from datasets import Dataset  # noqa: PLC0415
        from ragas import evaluate  # noqa: PLC0415
        from ragas.run_config import RunConfig  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "RAGAS or one of its optional dependencies could not be imported.\n"
            f"Original error: {exc}\n\n"
            "If RAGAS itself is missing, install it first, for example:\n"
            "  .\\backend\\.venv\\Scripts\\python.exe -m ensurepip --upgrade\n"
            "  .\\backend\\.venv\\Scripts\\python.exe -m pip install ragas datasets pandas\n"
            "If the error mentions langchain_community.chat_models.vertexai, this script already patches that optional import; "
            "rerun after pulling the latest script changes.\n"
            "Then rerun with --phase judge --judge ragas."
        )
        raise ImportError(msg) from exc
    metrics = _load_ragas_metrics(metric_names)
    run_config = RunConfig(
        timeout=timeout,
        max_retries=max_retries,
        max_workers=max_workers,
    )

    dataset_rows = [
        {
            "user_input": str(row["question"]),
            "response": str(row["answer"]),
            "retrieved_contexts": [str(value) for value in row.get("contexts", [])],
            "reference": str(row["ground_truth_answer"]),
        }
        for row in rows
    ]
    print(
        f"[judge/ragas] dataset rows={len(dataset_rows)} metrics={list(metric_names)} "
        f"max_workers={max_workers} max_retries={max_retries} timeout={timeout}",
        flush=True,
    )
    print("[judge/ragas] creating judge model...", flush=True)
    llm = _ragas_llm(_create_role_chat_model(config, role="judge", model_name=judge_model), run_config=run_config)
    embeddings = _ragas_embeddings(_create_langchain_embeddings(config))
    print("[judge/ragas] evaluate started; RAGAS may take several minutes...", flush=True)
    result = evaluate(
        Dataset.from_list(dataset_rows),
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )
    print("[judge/ragas] evaluate finished; formatting scores...", flush=True)
    score_records = _ragas_result_records(result)
    output_rows: list[dict[str, object]] = []
    for source, score in zip(rows, score_records, strict=False):
        output_rows.append(
            {
                "id": source["id"],
                "type": source["type"],
                "difficulty": source["difficulty"],
                "mode": source["mode"],
                "question": source["question"],
                "answer": source["answer"],
                "ground_truth_answer": source["ground_truth_answer"],
                "scores": _numeric_scores(score),
            }
        )
    return output_rows, _summary(output_rows)


def _parse_ragas_metric_names(raw: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(name.strip() for name in raw.split(",") if name.strip()))
    if not names:
        msg = "--ragas-metrics must contain at least one metric name"
        raise ValueError(msg)
    supported = {"faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"}
    unknown = sorted(set(names) - supported)
    if unknown:
        msg = f"Unsupported RAGAS metrics: {', '.join(unknown)}. Supported: {', '.join(sorted(supported))}"
        raise ValueError(msg)
    return names


def _load_ragas_metrics(metric_names: Sequence[str]) -> list[object]:
    from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness  # noqa: PLC0415

    registry = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }
    return [registry[name] for name in metric_names]


def _run_llm_judge(rows: Sequence[Mapping[str, object]], *, config: AgentRuntimeConfig, judge_model: str | None) -> tuple[list[dict[str, object]], dict[str, object]]:
    model = _create_role_chat_model(config, role="judge", model_name=judge_model)
    output_rows: list[dict[str, object]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        print(f"[judge/llm] {index}/{total} {row.get('id')} judging...", flush=True)
        scores = _invoke_json_judge(
            model,
            question=str(row["question"]),
            answer=str(row["answer"]),
            ground_truth=str(row["ground_truth_answer"]),
            contexts=[str(value) for value in row.get("contexts", [])],
        )
        output_rows.append(
            {
                "id": row["id"],
                "type": row["type"],
                "difficulty": row["difficulty"],
                "mode": row["mode"],
                "question": row["question"],
                "answer": row["answer"],
                "ground_truth_answer": row["ground_truth_answer"],
                "scores": scores,
            }
        )
        print(f"[judge/llm] {index}/{total} {row.get('id')} done overall={scores.get('overall')}", flush=True)
    return output_rows, _summary(output_rows)


def _create_role_chat_model(config: AgentRuntimeConfig, *, role: Literal["answer", "judge"], model_name: str | None) -> object:
    """Create an OpenAI-compatible chat model for answer generation or judging."""
    prefix = f"RAGAS_{role.upper()}"
    api_key = os.environ.get(f"{prefix}_API_KEY")
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    enable_thinking = _optional_bool(os.environ.get(f"{prefix}_ENABLE_THINKING"))
    if not api_key and not base_url and enable_thinking is None:
        return create_dashscope_model(config, model_name=model_name)

    try:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install langchain-openai before using role-specific RAGAS chat model settings."
        raise ImportError(msg) from exc
    resolved_api_key = api_key or config.dashscope_api_key
    if not resolved_api_key:
        msg = f"Set {prefix}_API_KEY or DASHSCOPE_API_KEY before creating the {role} model."
        raise ValueError(msg)
    extra_body = None
    if enable_thinking is not None:
        extra_body = {"enable_thinking": enable_thinking}
    elif config.dashscope_enable_thinking is not None:
        extra_body = {"enable_thinking": config.dashscope_enable_thinking}
    return ChatOpenAI(
        model=model_name or config.chat_model,
        api_key=resolved_api_key,
        base_url=base_url or config.dashscope_base_url,
        extra_body=extra_body,
    )


def _optional_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    msg = f"Invalid boolean value: {value!r}"
    raise ValueError(msg)


def _patch_ragas_optional_vertexai() -> None:
    """Patch a RAGAS optional VertexAI import that is unused in this script."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules or find_spec(module_name) is not None:
        return
    vertexai_module = types.ModuleType(module_name)

    class ChatVertexAI:  # noqa: D401 - compatibility stub only.
        """Compatibility stub for RAGAS imports when VertexAI is not used."""

    vertexai_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = vertexai_module


def _invoke_answer_model(model: object, *, question: str, contexts: Sequence[str]) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    context_block = "\n\n".join(f"[Context {index}]\n{text}" for index, text in enumerate(contexts, start=1))
    prompt = (
        f"Question:\n{question}\n\n"
        f"Contexts:\n{context_block or '(no retrieved context)'}\n\n"
        "Answer in Chinese. Use only the contexts. If key information is missing, say what is missing."
    )
    result = model.invoke([SystemMessage(content=_DEFAULT_SYSTEM_PROMPT), HumanMessage(content=prompt)])
    return _message_content(result).strip()


def _invoke_json_judge(model: object, *, question: str, answer: str, ground_truth: str, contexts: Sequence[str]) -> dict[str, float]:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    context_block = "\n\n".join(f"[Context {index}]\n{text}" for index, text in enumerate(contexts, start=1))
    prompt = (
        "Score this RAG answer from 0.0 to 1.0. Return strict JSON only with keys: "
        "answer_correctness, faithfulness, context_recall, context_precision, overall.\n\n"
        f"Question:\n{question}\n\nReference answer:\n{ground_truth}\n\nGenerated answer:\n{answer}\n\nContexts:\n{context_block}"
    )
    result = model.invoke(
        [
            SystemMessage(content="You are a strict RAG evaluator. Return only valid JSON."),
            HumanMessage(content=prompt),
        ]
    )
    raw = _message_content(result)
    payload = _parse_json_object(raw)
    return {
        "answer_correctness": _bounded_float(payload.get("answer_correctness")),
        "faithfulness": _bounded_float(payload.get("faithfulness")),
        "context_recall": _bounded_float(payload.get("context_recall")),
        "context_precision": _bounded_float(payload.get("context_precision")),
        "overall": _bounded_float(payload.get("overall")),
    }


def _resolve_scope(*, config: AgentRuntimeConfig, kb_id: str, tenant_id: str, user_id: str) -> tuple[str, str | None]:
    resolved_tenant = tenant_id
    resolved_user = user_id
    if config.postgres_dsn and (not resolved_tenant or not resolved_user):
        inferred_tenant, inferred_user = infer_scope_from_kb(dsn=config.postgres_dsn, kb_id=kb_id)
        resolved_tenant = resolved_tenant or inferred_tenant
        resolved_user = resolved_user or inferred_user
    return resolved_tenant or config.tenant_id, resolved_user or config.user_id


def _create_langchain_embeddings(config: AgentRuntimeConfig) -> object:
    try:
        from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install langchain-openai before running RAGAS embedding metrics."
        raise ImportError(msg) from exc
    return OpenAIEmbeddings(
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
        api_key=config.dashscope_api_key,
        base_url=config.dashscope_base_url,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )


def _ragas_llm(model: object, *, run_config: object | None = None) -> object:
    try:
        from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415
    except ImportError:
        return model
    return LangchainLLMWrapper(model, run_config=run_config, bypass_n=True)


def _ragas_embeddings(embeddings: object) -> object:
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415
    except ImportError:
        return embeddings
    return LangchainEmbeddingsWrapper(embeddings)


def _ragas_result_records(result: object) -> list[dict[str, object]]:
    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        return cast("list[dict[str, object]]", frame.to_dict(orient="records"))
    if isinstance(result, Mapping):
        return [dict(result)]
    scores = getattr(result, "scores", None)
    if isinstance(scores, list):
        return [dict(item) for item in scores if isinstance(item, Mapping)]
    return []


def _numeric_scores(row: Mapping[str, object]) -> dict[str, float]:
    ignored = {"user_input", "response", "retrieved_contexts", "reference", "question", "answer", "contexts", "ground_truth"}
    scores: dict[str, float] = {}
    for key, value in row.items():
        if key in ignored:
            continue
        try:
            scores[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    overall_values = [_scores(row) for row in rows]
    by_type_values: dict[str, list[Mapping[str, float]]] = {}
    for row in rows:
        by_type_values.setdefault(str(row.get("type") or "unknown"), []).append(_scores(row))
    return {
        "overall": aggregate_metric_dicts(overall_values),
        "by_type": {key: aggregate_metric_dicts(values) for key, values in sorted(by_type_values.items())},
    }


def _scores(row: Mapping[str, object]) -> Mapping[str, float]:
    scores = row.get("scores")
    if not isinstance(scores, Mapping):
        return {}
    return {str(key): float(value) for key, value in scores.items()}


def _retrieved_context_row(chunk: RetrievedChunk, *, rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.metadata.title,
        "chunk_index": chunk.metadata.chunk_index,
        "vector_score": chunk.vector_score,
        "keyword_score": chunk.keyword_score,
        "fused_score": chunk.fused_score,
        "rerank_score": chunk.rerank_score,
    }


def _context_text(chunk: RetrievedChunk) -> str:
    text = chunk.text.strip()
    if text:
        return text
    return " ".join(part for part in (chunk.metadata.title, chunk.metadata.section_path, chunk.metadata.source_uri) if part).strip()


def _message_content(message: object) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _parse_json_object(raw: str) -> dict[str, object]:
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _bounded_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _load_jsonl(path: Path | str) -> list[dict[str, object]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(cast("dict[str, object]", json.loads(line)))
    return rows


if __name__ == "__main__":
    main()
