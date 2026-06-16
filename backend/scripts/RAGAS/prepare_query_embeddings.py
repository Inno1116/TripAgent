"""Prepare cached query embeddings for the tourism RAG evaluation set."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_common import (
    default_embedding_file,
    default_eval_file,
    embedding_version,
    load_eval_cases,
    load_query_embeddings,
    load_runtime_env,
    normalize_query,
    write_jsonl,
)
from kyuriagents.runtime import AgentRuntimeConfig
from kyuriagents.runtime.dashscope import create_dashscope_embed_query


def main() -> None:
    """Embed evaluation questions once and store them as JSONL."""
    parser = argparse.ArgumentParser(description="Cache query embeddings for tourism RAG retrieval evaluation.")
    parser.add_argument("--eval-file", type=Path, default=default_eval_file(), help="Annotated tourism eval JSONL.")
    parser.add_argument("--output", type=Path, default=default_embedding_file(), help="Output query embedding JSONL.")
    parser.add_argument("--force", action="store_true", help="Re-embed every question even when a cache row exists.")
    args = parser.parse_args()

    load_runtime_env()
    config = AgentRuntimeConfig.from_env()
    cases = load_eval_cases(args.eval_file)
    existing = {} if args.force else load_query_embeddings(args.output)
    expected_version = embedding_version(model=config.embedding_model, dimensions=config.embedding_dimensions)
    embed_query = create_dashscope_embed_query(config)

    rows: list[dict[str, object]] = []
    embedded = 0
    reused = 0
    for case in cases:
        normalized_query = normalize_query(case.question)
        cached = existing.get(normalized_query)
        if (
            cached is not None
            and cached.embedding_model == config.embedding_model
            and cached.embedding_version == expected_version
        ):
            embedding = cached.embedding
            reused += 1
        else:
            embedding = tuple(float(value) for value in embed_query(normalized_query))
            embedded += 1
        rows.append(
            {
                "id": case.id,
                "query": case.question,
                "normalized_query": normalized_query,
                "embedding_model": config.embedding_model,
                "embedding_version": expected_version,
                "dimensions": len(embedding),
                "embedding": list(embedding),
            }
        )

    write_jsonl(args.output, rows)
    print(
        {
            "eval_file": str(args.eval_file),
            "output": str(args.output),
            "count": len(rows),
            "embedded": embedded,
            "reused": reused,
            "embedding_model": config.embedding_model,
            "embedding_version": expected_version,
        }
    )


if __name__ == "__main__":
    main()
