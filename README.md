# KyuriAgents

KyuriAgents 是一个面向个人知识库问答、长期记忆和任务规划的智能体应用平台。项目现在按业务形态整理为三个主要目录：

```text
backend/    FastAPI 后端、Agent 运行时、RAG、Memory、Task、WebSearch、Ingestion
frontend/   Next.js 前端
deploy/     Docker Compose、镜像构建和部署脚本
```

后端已经不再依赖原 DeepAgent 的 `create_deep_agent` 图工厂，而是通过 LangChain/LangGraph `create_agent` 直接组装模型、工具和中间件。旧的 `libs/` 目录暂时保留用于对照和回滚，后续确认新结构稳定后再清理。

## 核心能力

- 用户注册登录、多轮对话、流式响应
- 文档上传与个人知识库构建
- Elasticsearch + Milvus 混合 RAG 检索
- PostgreSQL 长期记忆与线程摘要
- Redis Pending Turn，避免异常请求污染正式对话历史
- 任务规划与执行，支持步骤观察、重试、Replan 和用户补充信息
- SearXNG 搜索、静态网页抓取、Playwright 动态渲染
- 工具治理、超时、审计和风险分类
- Docker 一键部署 PostgreSQL、Redis、Elasticsearch、Milvus、SearXNG、API、Worker、Web

## 后端启动

本地复用当前虚拟环境：

```powershell
$env:PYTHONPATH = (Resolve-Path .\backend).Path
.\libs\deepagents\.venv\Scripts\python.exe backend\scripts\api_server.py
```

或安装独立后端包：

```powershell
cd backend
python -m pip install -e .
python scripts\api_server.py
```

后端仍兼容已有 `DEEPAGENTS_*` 环境变量，同时支持新的 `KYURIAGENTS_*` 前缀别名。核心配置包括：

- `DASHSCOPE_API_KEY`
- `DEEPAGENTS_POSTGRES_DSN` 或 `KYURIAGENTS_POSTGRES_DSN`
- `DEEPAGENTS_REDIS_URL` 或 `KYURIAGENTS_REDIS_URL`
- `RAG_ES_URL`
- `RAG_MILVUS_URI`
- `SEARXNG_BASE_URL`

## 前端启动

```powershell
cd frontend
npm install
npm run dev
```

默认开发地址：

```text
http://127.0.0.1:5173
```

## 部署

完整 Docker 部署：

```powershell
cd deploy
.\deploy.ps1 init
# 编辑 runtime.env，填写 DashScope Key、PostgreSQL 密码、管理员密钥等
.\deploy.ps1 up
```

常用命令：

```powershell
.\deploy.ps1 update
.\deploy.ps1 logs api
.\deploy.ps1 logs worker
.\deploy.ps1 down
```

## 验证

本次整理后已经验证：

- 后端 `kyuriagents` 包可直接从 `backend/` 导入
- `create_kyuri_agent` 不再调用 `create_deep_agent`
- 后端关键文件通过 ruff 检查
- 后端源码通过 Python 编译检查
- 前端 `npm run typecheck` 通过
- 前端 `npm run build` 通过
- `docker-compose.full.yml` 和 `docker-compose.external.yml` 均可解析

作者联系方式：210825684@qq.com
