# KyuriAgents 部署说明

`deploy/` 用于把 KyuriAgents 部署到服务器。推荐先使用完整 Docker Compose 方案，等项目稳定后再把 PostgreSQL、Elasticsearch、Milvus 拆到云厂商托管服务。

## 完整服务

完整部署会启动：

- Web 前端
- FastAPI 后端
- 文档解析 Worker
- PostgreSQL
- Redis
- Elasticsearch
- Milvus，以及 Milvus 依赖的 etcd 和 MinIO
- SearXNG
- Bootstrap 初始化任务

## 快速开始

Linux:

```bash
cd deploy
bash deploy.sh init
```

Windows PowerShell:

```powershell
cd deploy
.\deploy.ps1 init
```

第一次执行会生成 `runtime.env`。至少修改：

```env
DASHSCOPE_API_KEY=你的 DashScope API Key
POSTGRES_PASSWORD=强密码
MINIO_ROOT_PASSWORD=强密码
KYURIAGENTS_API_ADMIN_KEY=随机管理密钥
KYURIAGENTS_POSTGRES_DSN=postgresql://kyuriagents:同一个Postgres密码@postgres:5432/kyuriagents
```

启动：

```bash
bash deploy.sh up
```

或：

```powershell
.\deploy.ps1 up
```

默认访问：

```text
http://服务器IP:8080
```

## 常用命令

```bash
bash deploy.sh update
bash deploy.sh logs api
bash deploy.sh logs worker
bash deploy.sh ps
bash deploy.sh down
```

PowerShell 对应：

```powershell
.\deploy.ps1 update
.\deploy.ps1 logs api
.\deploy.ps1 logs worker
.\deploy.ps1 ps
.\deploy.ps1 down
```

彻底删除容器和数据卷需要显式确认：

```bash
KYURIAGENTS_CONFIRM_RESET=yes bash deploy.sh reset
```

```powershell
$env:KYURIAGENTS_CONFIRM_RESET="yes"
.\deploy.ps1 reset
```

## 外部数据库模式

如果 PostgreSQL、Elasticsearch、Milvus 已经单独部署，可以使用：

```bash
KYURIAGENTS_COMPOSE_FILE=docker-compose.external.yml bash deploy.sh up
```

此模式仍会启动 Redis、SearXNG、API、Worker、Web。你需要在 `runtime.env` 中配置：

```env
KYURIAGENTS_POSTGRES_DSN=postgresql://用户:密码@PostgreSQL地址:5432/kyuriagents
RAG_ES_URL=http://Elasticsearch地址:9200
RAG_MILVUS_URI=http://Milvus地址:19530
```

## 端口建议

生产环境建议只对公网暴露：

```text
8080  Web
```

不要直接暴露：

```text
5432   PostgreSQL
6379   Redis
9200   Elasticsearch
19530  Milvus
9000   MinIO
2379   etcd
```

Linux 上 Elasticsearch 通常需要：

```bash
sudo sysctl -w vm.max_map_count=262144
```

## 文档上传

当前 Worker 支持：

- PDF：文本型 PDF
- DOCX：现代 Word 文档
- TXT：纯文本

解析模式：

```env
KYURIAGENTS_INGESTION_PARSER=auto
```

- `local`：本地解析
- `mcp`：调用 MCP 解析工具
- `auto`：配置 MCP 时优先 MCP，否则本地解析
