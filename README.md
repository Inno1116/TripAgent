# Kyuri:TripAgent

Kyuri:TripAgent 是一个面向旅行规划场景的智能体应用平台。系统支持用户通过自然语言提出出行需求，例如目的地、天数、预算、兴趣偏好、交通限制等，并由 Agent 结合长期记忆、个人知识库、实时网页检索和任务规划能力，生成结构化旅行方案。

项目目标不是只做一个聊天窗口，而是构建一个可部署、可扩展、可观察的旅行规划 Agent 工作台：

- 支持注册登录、多轮对话和流式响应
- 支持长期记忆，记录用户偏好、预算习惯、旅行风格等稳定信息
- 支持上传文档构建个人知识库，例如攻略、PDF 行程单、景点资料
- 支持 Elasticsearch + Milvus 混合 RAG 检索
- 支持 SearXNG + Playwright 的联网搜索与网页阅读
- 支持任务规划，将复杂旅行需求拆解为可执行步骤
- 支持 PostgreSQL、Redis、Elasticsearch、Milvus、SearXNG 的 Docker 部署
- 支持前端展示任务阶段、执行步骤、工具结果和最终方案

## 技术栈

```text
Frontend:   Next.js, TypeScript, React
Backend:    FastAPI, Python
Agent:      LangChain, LangGraph
Database:   PostgreSQL, Redis
Retrieval:  Elasticsearch, Milvus
WebSearch:  SearXNG, Playwright
LLM:        DashScope compatible chat / embedding / rerank models
Deploy:     Docker Compose
```

## 目录结构

```text
backend/    后端服务、Agent 运行时、RAG、Memory、Task、WebSearch、Ingestion
frontend/   Next.js 前端
deploy/     Docker Compose、镜像构建和部署脚本
```

## 核心功能

### 1. 智能旅行规划

用户可以用自然语言提出复杂旅行需求：

```text
帮我规划一个 4 天 3 晚的大阪 + 京都旅行计划，预算中等，喜欢动漫、寺庙和美食。
```

Agent 会根据需求生成旅行计划，通常包括：

- 行程总览
- 每日安排
- 景点与美食推荐
- 交通建议
- 预算参考
- 注意事项
- 可选替代方案

### 2. 任务规划与执行

复杂请求会进入任务模式：

```text
User Query
  -> Context Builder
  -> Planner
  -> Plan Validator
  -> Step Executor
  -> Observer
  -> Router
  -> Replan / Ask User / Answer
```

系统会把目标拆解为 `think / web / rag / process / answer` 等步骤，逐步执行并记录状态。如果信息不足，会请求用户补充；如果工具失败，会进入重试或重规划流程。

### 3. 混合 RAG 知识库

用户可以上传 PDF、DOCX、TXT 等文档构建个人知识库。文档处理流程：

```text
上传文件
  -> 写入 PostgreSQL 文档与任务记录
  -> Redis 唤醒 Worker
  -> 文档解析
  -> 分块
  -> Embedding
  -> 写入 Elasticsearch + Milvus
  -> 更新任务状态
```

检索时使用：

- Elasticsearch：关键词和稀疏检索
- Milvus：向量语义检索
- Rerank：对候选结果重新排序
- PostgreSQL：保存权威文档、知识库、任务元数据

### 4. 长期记忆与上下文管理

长期记忆用于保存用户跨会话稳定信息，例如：

- 喜欢的旅行风格
- 预算偏好
- 饮食禁忌
- 常用出发城市
- 住宿偏好

上下文管理基于 token 预算，在长对话中保留近期消息，并将较早历史压缩为摘要，避免上下文无限增长。

### 5. 联网搜索与网页阅读

联网搜索基于 SearXNG，支持：

- 查询规划
- 多查询搜索
- URL 去重
- 结果重排
- 静态网页抓取
- Playwright 动态渲染
- 抓取失败诊断

适合获取实时旅行信息，例如景点开放时间、城市活动、交通变化、签证政策、天气相关信息等。

### 6. 安全与稳定性

系统包含多项安全和稳定性处理：

- 用户输入 token 上限
- Redis Pending Turn，避免异常请求污染正式历史
- 工具超时与失败处理
- 工具调用审计
- 用户、租户、知识库隔离
- 文档上传大小限制
- 任务状态持久化
- Docker 服务隔离

## 本地开发

### 1. 启动后端

如果你已经在 `backend/` 下创建了独立虚拟环境：

```powershell
cd D:\Agent\Kyuriagents\backend
.\.venv\Scripts\python.exe .\scripts\api_server.py
```

如果需要重新创建虚拟环境：

```powershell
cd D:\Agent\Kyuriagents\backend

$uv = "$env:USERPROFILE\.local\bin\uv.exe"
& $uv venv .venv --python 3.11
& $uv pip install --python .\.venv\Scripts\python.exe -e .
& .\.venv\Scripts\python.exe -m playwright install chromium
```

启动：

```powershell
.\.venv\Scripts\python.exe .\scripts\api_server.py
```

后端默认地址：

```text
http://127.0.0.1:8000
```

### 2. 启动 Worker

如果需要测试文档上传与知识库构建，需要单独启动 Worker：

```powershell
cd D:\Agent\Kyuriagents\backend
.\.venv\Scripts\python.exe .\scripts\ingestion_worker.py
```

### 3. 启动前端

```powershell
cd D:\Agent\Kyuriagents\frontend
npm install
npm run dev
```

前端默认地址：

```text
http://127.0.0.1:5173
```

## 环境变量

核心环境变量包括：

```env
DASHSCOPE_API_KEY=replace-me
DASHSCOPE_CHAT_MODEL=qwen-plus
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_EMBEDDING_DIMENSIONS=1024

KYURIAGENTS_POSTGRES_DSN=postgresql://user:password@localhost:5432/kyuriagents
KYURIAGENTS_REDIS_URL=redis://localhost:6379/0

RAG_ES_URL=http://localhost:9200
RAG_ES_INDEX=rag_chunks
RAG_MILVUS_URI=http://localhost:19530
RAG_MILVUS_COLLECTION=rag_chunks

KYURIAGENTS_ENABLE_RAG=true
KYURIAGENTS_ENABLE_MEMORY=true
KYURIAGENTS_ENABLE_WEB_SEARCH=true
SEARXNG_BASE_URL=http://127.0.0.1:8888
```

部署示例见：

```text
deploy/runtime.env.example
```

## Docker 部署

进入部署目录：

```powershell
cd D:\Agent\Kyuriagents\deploy
```

生成配置：

```powershell
.\deploy.ps1 init
```

编辑 `runtime.env`，至少填写：

```env
DASHSCOPE_API_KEY=你的 DashScope API Key
POSTGRES_PASSWORD=强密码
MINIO_ROOT_PASSWORD=强密码
KYURIAGENTS_API_ADMIN_KEY=随机管理密钥
KYURIAGENTS_POSTGRES_DSN=postgresql://kyuriagents:同一个Postgres密码@postgres:5432/kyuriagents
```

启动完整服务：

```powershell
.\deploy.ps1 up
```

访问：

```text
http://127.0.0.1:8080
```

常用命令：

```powershell
.\deploy.ps1 update
.\deploy.ps1 logs api
.\deploy.ps1 logs worker
.\deploy.ps1 ps
.\deploy.ps1 down
```

## 外部依赖部署

如果 PostgreSQL、Elasticsearch、Milvus 已经部署在外部，可以使用：

```powershell
$env:KYURIAGENTS_COMPOSE_FILE="docker-compose.external.yml"
.\deploy.ps1 up
```

然后在 `runtime.env` 中配置外部服务地址：

```env
KYURIAGENTS_POSTGRES_DSN=postgresql://user:password@postgres-host:5432/kyuriagents
RAG_ES_URL=http://elasticsearch-host:9200
RAG_MILVUS_URI=http://milvus-host:19530
```

## 验证命令

后端基础检查：

```powershell
cd D:\Agent\Kyuriagents\backend
.\.venv\Scripts\python.exe -m compileall -q kyuriagents scripts
```

前端检查：

```powershell
cd D:\Agent\Kyuriagents\frontend
npm run typecheck
npm run build
```

Compose 配置检查：

```powershell
cd D:\Agent\Kyuriagents\deploy
$env:KYURIAGENTS_ENV_FILE="runtime.env.example"
docker compose --env-file runtime.env.example -f docker-compose.full.yml config --quiet
```
