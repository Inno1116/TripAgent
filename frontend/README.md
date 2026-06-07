# KyuriAgents Frontend

这是 KyuriAgents 的 Next.js + TypeScript 前端。

## 启动

先启动后端：

```powershell
$env:PYTHONPATH = (Resolve-Path ..\backend).Path
..\libs\deepagents\.venv\Scripts\python.exe ..\backend\scripts\api_server.py
```

再启动前端：

```powershell
npm install
npm run dev
```

打开：

```text
http://127.0.0.1:5173
```

默认 API 地址是 `http://127.0.0.1:8000`。如需修改，复制 `.env.example` 为 `.env.local` 后设置：

```text
NEXT_PUBLIC_KYURIAGENTS_API_BASE=http://127.0.0.1:8000
```
