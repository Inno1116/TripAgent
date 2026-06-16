# Beijing Tourism RAG Evaluation

This folder contains an isolated retrieval evaluation workflow for the Beijing
tourism knowledge-base PDFs. It does not change the runtime API or ingestion
pipeline.

## Files

- `tourism_eval.jsonl`: manually annotated questions, answers, and gold chunks.
- `prepare_query_embeddings.py`: embeds each question once and writes a reusable
  query embedding cache.
- `tourism_retrieval_eval.py`: compares `vector`, `hybrid`, and
  `hybrid_rerank` retrieval modes with Recall/MRR/NDCG.
- `tourism_ragas_eval.py`: generates end-to-end RAG answers and evaluates them
  with RAGAS or a compact LLM judge.
- `tourism_query_embeddings.jsonl`: generated query embedding cache.
- `tourism_retrieval_results.jsonl`: generated per-question retrieval results.
- `tourism_ragas_generations.jsonl`: generated answers and retrieved contexts.
- `tourism_ragas_scores.jsonl`: generated RAGAS scores.

## Run

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\backend).Path
```

Create a local evaluation env file when you want RAGAS to use a separate
DashScope key/model from the running backend service:

```powershell
Copy-Item .\backend\scripts\RAGAS\RAGAS.env.example .\backend\scripts\RAGAS\RAGAS.env
```

Then edit `backend/scripts/RAGAS/RAGAS.env`. The scripts load
`backend/runtime.env` as a fallback, then load `RAGAS.env` last and override
duplicate variables. For example:

```env
DASHSCOPE_API_KEY=your_eval_key
DASHSCOPE_CHAT_MODEL=qwen-plus
DASHSCOPE_ENABLE_THINKING=false
RAGAS_ANSWER_MODEL=
RAGAS_JUDGE_MODEL=
```

If your DashScope account can use another OpenAI-compatible chat model, set it
as `DASHSCOPE_CHAT_MODEL` in `RAGAS.env`.

To use a separate judge model without changing answer generation, set:

```env
RAGAS_JUDGE_MODEL=qwen-plus
```

If the judge is served by another OpenAI-compatible provider such as DeepSeek,
also set:

```env
RAGAS_JUDGE_API_KEY=your_judge_key
RAGAS_JUDGE_BASE_URL=https://api.deepseek.com
RAGAS_JUDGE_MODEL=deepseek-chat
```

Prepare query embeddings once:

```powershell
.\backend\.venv\Scripts\python.exe .\backend\scripts\RAGAS\prepare_query_embeddings.py
```

Run retrieval evaluation:

```powershell
.\backend\.venv\Scripts\python.exe .\backend\scripts\RAGAS\tourism_retrieval_eval.py --kb-id kb_2bf22e469fd2403989782566b1180393
```

To avoid rerank API cost during debugging:

```powershell
.\backend\.venv\Scripts\python.exe .\backend\scripts\RAGAS\tourism_retrieval_eval.py --kb-id kb_2bf22e469fd2403989782566b1180393 --modes vector,hybrid
```

## Metrics

- `recall_at_3`: fraction of gold chunks retrieved in top 3.
- `recall_at_k`: fraction of gold chunks retrieved in final top K.
- `mrr_at_k`: reciprocal rank of the first retrieved gold chunk.
- `ndcg_at_k`: binary-relevance ranking quality for gold chunks.

The default `top-k` is 5.

## End-to-End RAGAS / LLM Judge

Generate answers and run official RAGAS metrics:

```powershell
.\backend\.venv\Scripts\python.exe .\backend\scripts\RAGAS\tourism_ragas_eval.py `
  --kb-id kb_2bf22e469fd2403989782566b1180393 `
  --mode hybrid_rerank `
  --judge ragas `
  --ragas-metrics faithfulness,answer_correctness,context_recall `
  --ragas-max-workers 1 `
  --ragas-max-retries 1
```

For a cheap smoke test, only run the first 3 questions:

```powershell
.\backend\.venv\Scripts\python.exe .\backend\scripts\RAGAS\tourism_ragas_eval.py `
  --kb-id kb_2bf22e469fd2403989782566b1180393 `
  --mode hybrid_rerank `
  --judge ragas `
  --limit 3 `
  --ragas-metrics faithfulness,answer_correctness,context_recall `
  --ragas-max-workers 1 `
  --ragas-max-retries 1
```

The script is resumable. To split the workflow:

```powershell
.\backend\.venv\Scripts\python.exe .\backend\scripts\RAGAS\tourism_ragas_eval.py `
  --kb-id kb_2bf22e469fd2403989782566b1180393 `
  --mode hybrid_rerank `
  --phase generate

.\backend\.venv\Scripts\python.exe .\backend\scripts\RAGAS\tourism_ragas_eval.py `
  --phase judge `
  --judge ragas
```

If RAGAS is not installed, install it first:

```powershell
.\backend\.venv\Scripts\python.exe -m ensurepip --upgrade
.\backend\.venv\Scripts\python.exe -m pip install ragas datasets pandas
```

If you only want a lightweight project-local judge without installing RAGAS:

```powershell
.\backend\.venv\Scripts\python.exe .\backend\scripts\RAGAS\tourism_ragas_eval.py `
  --kb-id kb_2bf22e469fd2403989782566b1180393 `
  --mode hybrid_rerank `
  --judge llm
```
