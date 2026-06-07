"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { BookOpen, Bot, FileText, Globe2, LogOut, Menu, MessageSquare, Plus, Send, Settings, Trash2, Upload } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { KyuriAvatar } from "@/components/workspace/kyuri-avatar";
import { apiRequest, binaryRequest, streamRequest } from "@/lib/api";
import { removeKeys, readJson, readString, STORAGE, writeString } from "@/lib/storage";
import type {
  AuthMode,
  AuthResponse,
  DocumentRecord,
  KnowledgeBaseRecord,
  MessageRecord,
  ParserMode,
  RecordResponse,
  TaskEventRecord,
  TaskRecord,
  TaskSnapshotPayload,
  ThreadRecord,
  UserRecord,
  View,
} from "@/lib/types";
import { cn, displayDate, titleFromMessage } from "@/lib/utils";

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_KYURIAGENTS_API_BASE || "http://127.0.0.1:8000";
const AUTHOR_EMAIL = "210825684@qq.com";

let nextClientMessageId = 1;

function createClientMessageId() {
  const value = `client-message-${Date.now()}-${nextClientMessageId}`;
  nextClientMessageId += 1;
  return value;
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "请求失败。";
}

function mergeTaskEvents(current: TaskEventRecord[] = [], incoming: TaskEventRecord[] = []) {
  const merged = new Map<string, TaskEventRecord>();
  for (const event of current) merged.set(event.event_id, event);
  for (const event of incoming) merged.set(event.event_id, event);
  return Array.from(merged.values()).sort((left, right) => String(left.created_at || "").localeCompare(String(right.created_at || "")));
}

function taskEventStatus(event?: TaskEventRecord) {
  if (!event) return "正在执行任务...";
  if (event.event_type === "intent") return "已判断任务意图";
  if (event.event_type === "context") return "已整理上下文";
  if (event.event_type === "planned") return "计划已生成";
  if (event.event_type === "validated") return "计划已校验";
  if (event.event_type === "hitl_requested") return "需要你补充信息";
  if (event.event_type === "hitl_resumed") return "已收到补充，继续任务";
  if (event.event_type === "step_started") return `正在执行：${event.message}`;
  if (event.event_type === "step_finished") return `已完成：${event.message}`;
  if (event.event_type === "retry") return "工具失败，正在重试";
  if (event.event_type === "replanned") return "已重新规划";
  if (event.event_type === "skipped") return "已跳过不可用步骤";
  if (event.event_type === "failed") return "任务执行失败";
  if (event.event_type === "finished") return "任务已完成";
  return event.message || "任务状态更新";
}

function taskStatusLabel(status?: string) {
  if (status === "queued") return "排队";
  if (status === "planning") return "规划中";
  if (status === "waiting_user") return "等待补充";
  if (status === "running") return "执行中";
  if (status === "succeeded") return "完成";
  if (status === "failed") return "失败";
  if (status === "cancelled") return "取消";
  return status || "未知";
}

function stepStatusLabel(status?: string) {
  if (status === "pending") return "待执行";
  if (status === "running") return "进行中";
  if (status === "succeeded") return "完成";
  if (status === "failed") return "失败";
  if (status === "skipped") return "跳过";
  if (status === "cancelled") return "取消";
  return status || "待执行";
}

function stepStatusClass(status?: string) {
  if (status === "succeeded") return "border-emerald-200 bg-emerald-50 text-emerald-800";
  if (status === "running") return "border-violet-200 bg-violet-50 text-violet-800";
  if (status === "failed") return "border-rose-200 bg-rose-50 text-rose-800";
  if (status === "skipped") return "border-amber-200 bg-amber-50 text-amber-800";
  return "border-stone-200 bg-stone-50 text-stone-600";
}

function stepKindLabel(kind?: string) {
  if (kind === "rag") return "知识库";
  if (kind === "web") return "网页";
  if (kind === "process") return "分析";
  if (kind === "tool") return "工具";
  if (kind === "think") return "思考";
  if (kind === "answer") return "回答";
  return kind || "步骤";
}

function latestWaitingTask(items: MessageRecord[]) {
  for (let index = items.length - 1; index >= 0; index -= 1) {
    const task = items[index]?.task;
    if (task?.task_id && task.status === "waiting_user") return task;
  }
  return null;
}

function taskFlowStatusText(task?: TaskRecord | null) {
  return task?.status === "waiting_user" ? "正在补充任务信息..." : "正在规划任务...";
}

function stepOutputPreview(output?: string, maxChars = 360) {
  const text = (output || "").trim();
  if (!text) return "";
  return text.length > maxChars ? `${text.slice(0, maxChars)}...` : text;
}

export default function Home() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [token, setToken] = useState("");
  const [user, setUser] = useState<UserRecord | null>(null);
  const [threadId, setThreadId] = useState("");
  const [threads, setThreads] = useState<ThreadRecord[]>([]);
  const [messages, setMessages] = useState<MessageRecord[]>([]);
  const [activeView, setActiveViewState] = useState<View>("chat");
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [ragEnabled, setRagEnabledState] = useState(true);
  const [webSearchEnabled, setWebSearchEnabledState] = useState(true);
  const [taskMode, setTaskModeState] = useState(false);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBaseRecord[]>([]);
  const [selectedKbId, setSelectedKbId] = useState("");
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [knowledgeLoaded, setKnowledgeLoaded] = useState(false);
  const [knowledgeLoading, setKnowledgeLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [parserMode, setParserModeState] = useState<ParserMode>("auto");
  const [settingsStatus, setSettingsStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesRef = useRef<HTMLDivElement | null>(null);

  const selectedKb = useMemo(() => knowledgeBases.find((item) => item.kb_id === selectedKbId), [knowledgeBases, selectedKbId]);
  const currentThread = useMemo(() => threads.find((item) => item.thread_id === threadId), [threads, threadId]);
  const waitingTask = useMemo(() => latestWaitingTask(messages), [messages]);
  const email = user?.email || "已登录";

  useEffect(() => {
    const savedApiBase = readString(STORAGE.apiBase, DEFAULT_API_BASE) || DEFAULT_API_BASE;
    const savedToken = readString(STORAGE.token);
    setApiBase(savedApiBase);
    setToken(savedToken);
    setUser(readJson<UserRecord>(STORAGE.user));
    setThreadId(readString(STORAGE.threadId));
    setRagEnabledState(readString(STORAGE.ragEnabled, "true") !== "false");
    setWebSearchEnabledState(readString(STORAGE.webSearchEnabled, "true") !== "false");
    setTaskModeState(readString(STORAGE.taskMode) === "true");
    setActiveViewState((readString(STORAGE.activeView, "chat") as View) || "chat");
    setParserModeState((readString(STORAGE.ingestionParserMode, "auto") as ParserMode) || "auto");
    if (savedToken) {
      void bootstrap(savedApiBase, savedToken);
    }
  }, []);

  useEffect(() => {
    if (activeView === "knowledge" && token && !knowledgeLoaded && !knowledgeLoading) {
      void loadKnowledgeBases();
    }
  }, [activeView, knowledgeLoaded, knowledgeLoading, token]);

  useEffect(() => {
    if (activeView === "chat") {
      scrollMessagesToBottom();
    }
  }, [messages, activeView]);

  async function request<T>(path: string, options: RequestInit = {}) {
    return apiRequest<T>(apiBase, path, { ...options, token });
  }

  async function bootstrap(base: string, rawToken: string) {
    try {
      const me = await apiRequest<RecordResponse<{ user?: UserRecord }>>(base, "/v1/me", { token: rawToken });
      setUser(me.data.user || null);
      if (me.data.user) {
        window.localStorage.setItem(STORAGE.user, JSON.stringify(me.data.user));
      }
      const list = await apiRequest<RecordResponse<{ threads: ThreadRecord[] }>>(base, "/v1/threads", { token: rawToken });
      setThreads(list.data.threads || []);
      const savedThreadId = readString(STORAGE.threadId);
      if (savedThreadId) {
        await selectThread(savedThreadId, { base, rawToken });
      }
    } catch {
      clearSession();
    }
  }

  function writeSession(accessToken: string, data: AuthResponse["data"]) {
    setToken(accessToken);
    setUser(data.user || null);
    writeString(STORAGE.token, accessToken);
    writeString(STORAGE.user, JSON.stringify(data.user || null));
  }

  function clearSession() {
    setToken("");
    setUser(null);
    setThreadId("");
    setThreads([]);
    setMessages([]);
    removeKeys(STORAGE.token, STORAGE.user, STORAGE.threadId);
  }

  function persistApiBase(value: string) {
    const resolved = value.trim() || DEFAULT_API_BASE;
    setApiBase(resolved);
    writeString(STORAGE.apiBase, resolved);
  }

  function setActiveView(view: View) {
    setActiveViewState(view);
    setSidebarOpen(false);
    writeString(STORAGE.activeView, view);
  }

  function setRagEnabled(value: boolean) {
    setRagEnabledState(value);
    writeString(STORAGE.ragEnabled, String(value));
  }

  function setWebSearchEnabled(value: boolean) {
    setWebSearchEnabledState(value);
    writeString(STORAGE.webSearchEnabled, String(value));
  }

  function setTaskMode(value: boolean) {
    setTaskModeState(value);
    writeString(STORAGE.taskMode, String(value));
  }

  function setParserMode(value: ParserMode) {
    setParserModeState(value);
    writeString(STORAGE.ingestionParserMode, value);
  }

  async function refreshThreads() {
    const payload = await request<RecordResponse<{ threads: ThreadRecord[] }>>("/v1/threads");
    setThreads(payload.data.threads || []);
  }

  async function selectThread(id: string, override?: { base: string; rawToken: string }) {
    const base = override?.base || apiBase;
    const rawToken = override?.rawToken || token;
    setThreadId(id);
    setActiveViewState("chat");
    writeString(STORAGE.threadId, id);
    writeString(STORAGE.activeView, "chat");
    setError("");
    const payload = await apiRequest<RecordResponse<{ messages: MessageRecord[] }>>(base, `/v1/threads/${encodeURIComponent(id)}/messages`, {
      token: rawToken,
    });
    setMessages(payload.data.messages || []);
  }

  function startNewThread() {
    setThreadId("");
    setMessages([]);
    setError("");
    setActiveView("chat");
    removeKeys(STORAGE.threadId);
  }

  async function deleteThread(id: string) {
    if (!id || loading) return;
    const previousThreads = threads;
    const previousThreadId = threadId;
    const previousMessages = messages;
    setThreads((items) => items.filter((thread) => thread.thread_id !== id));
    if (previousThreadId === id) {
      setThreadId("");
      setMessages([]);
      removeKeys(STORAGE.threadId);
    }
    try {
      await request<RecordResponse<{ deleted: boolean; thread_id: string }>>(`/v1/threads/${encodeURIComponent(id)}`, { method: "DELETE" });
    } catch (caught) {
      setThreads(previousThreads);
      setThreadId(previousThreadId);
      setMessages(previousMessages);
      setError(errorMessage(caught));
    }
  }

  async function handleAuthSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const body: Record<string, string> = {
      tenant_id: String(form.get("tenant") || "default").trim() || "default",
      email: String(form.get("email") || ""),
      password: String(form.get("password") || ""),
    };
    if (authMode === "register") {
      body.display_name = String(form.get("displayName") || "");
    }
    setLoading(true);
    setError("");
    try {
      const path = authMode === "login" ? "/v1/auth/login" : "/v1/auth/register";
      const payload = await apiRequest<AuthResponse>(apiBase, path, { method: "POST", body: JSON.stringify(body) });
      writeSession(payload.access_token, payload.data);
      const list = await apiRequest<RecordResponse<{ threads: ThreadRecord[] }>>(apiBase, "/v1/threads", { token: payload.access_token });
      setThreads(list.data.threads || []);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setLoading(false);
    }
  }

  async function logout() {
    setLoading(true);
    try {
      await request<RecordResponse<{ revoked: boolean }>>("/v1/auth/logout", { method: "POST", body: "{}" });
    } catch {
      // Local logout should still succeed if the server-side token already expired.
    } finally {
      clearSession();
      setLoading(false);
    }
  }

  function updateMessage(clientId: string, update: (message: MessageRecord) => MessageRecord) {
    setMessages((items) => items.map((item) => (item.client_id === clientId ? update(item) : item)));
  }

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const text = String(form.get("message") || "").trim();
    if (!text || loading) return;
    event.currentTarget.reset();

    const pendingTask = waitingTask;
    const useTaskFlow = taskMode || Boolean(pendingTask);
    const initialStatus = useTaskFlow ? taskFlowStatusText(pendingTask) : "正在思考...";
    const optimistic: MessageRecord = { role: "user", content: text, client_id: createClientMessageId() };
    const assistantId = createClientMessageId();
    const assistant: MessageRecord = {
      role: "assistant",
      content: "",
      client_id: assistantId,
      phase: "thinking",
      status: initialStatus,
      receivedDelta: false,
    };
    setMessages((items) => [...items, optimistic, assistant]);
    setLoading(true);
    setStatusText(initialStatus);
    setError("");

    try {
      if (useTaskFlow) {
        await runTaskMessage(text, assistantId, pendingTask);
      } else {
        await runStreamMessage(text, assistantId);
      }
      await refreshThreads();
    } catch (caught) {
      updateMessage(assistantId, (message) => ({ ...message, phase: "error", status: errorMessage(caught) }));
      setError(errorMessage(caught));
    } finally {
      setLoading(false);
      setStatusText("");
    }
  }

  async function runStreamMessage(text: string, assistantId: string) {
    await streamRequest(
      apiBase,
      "/v1/chat/stream",
      {
        method: "POST",
        token,
        body: JSON.stringify({
          message: text,
          thread_id: threadId || null,
          title: threadId ? "" : titleFromMessage(text),
          rag_enabled: ragEnabled,
          web_search_enabled: webSearchEnabled,
        }),
      },
      {
        message_start(payload) {
          if (payload.thread_id) {
            setThreadId(payload.thread_id);
            writeString(STORAGE.threadId, payload.thread_id);
          }
        },
        status(payload) {
          const textStatus = payload.text || "正在调用工具...";
          setStatusText(textStatus);
          updateMessage(assistantId, (message) => ({ ...message, phase: "tool", status: textStatus }));
        },
        delta(payload) {
          const delta = payload.text || "";
          if (!delta) return;
          setStatusText("正在回复...");
          updateMessage(assistantId, (message) => ({
            ...message,
            content: `${message.content || ""}${delta}`,
            phase: "streaming",
            status: "",
            receivedDelta: true,
          }));
        },
        done(payload) {
          if (payload.thread_id) {
            setThreadId(payload.thread_id);
            writeString(STORAGE.threadId, payload.thread_id);
          }
          updateMessage(assistantId, (message) => ({
            ...message,
            message_id: payload.message_id || message.message_id,
            content: payload.content && (!message.receivedDelta || payload.replace === true) ? payload.content : message.content,
            phase: "done",
            status: "",
          }));
          setStatusText("");
        },
        error(payload) {
          throw new Error(payload.detail || "流式响应失败。");
        },
      },
    );
  }

  async function runTaskMessage(text: string, assistantId: string, pendingTask: TaskRecord | null = null) {
    const pendingTaskId = pendingTask?.task_id || "";
    const isResume = Boolean(pendingTaskId);
    const path = isResume ? `/v1/tasks/${encodeURIComponent(pendingTaskId)}/resume/stream` : "/v1/tasks/stream";
    const status = isResume ? "正在补充任务信息..." : "正在规划任务...";
    const body = isResume
      ? {
          message: text,
          intent: "task",
          rag_enabled: ragEnabled,
          web_search_enabled: webSearchEnabled,
        }
      : {
          goal: text,
          thread_id: threadId || null,
          title: threadId ? "" : titleFromMessage(text),
          intent: "task",
          rag_enabled: ragEnabled,
          web_search_enabled: webSearchEnabled,
        };
    setStatusText(status);
    updateMessage(assistantId, (message) => ({ ...message, phase: "thinking", status }));
    await streamRequest(
      apiBase,
      path,
      {
        method: "POST",
        token,
        body: JSON.stringify(body),
      },
      {
        task_start(payload) {
          applyTaskSnapshot(assistantId, payload, isResume ? "任务已继续" : "任务已创建");
        },
        task_event(payload) {
          const event = payload.event;
          const textStatus = taskEventStatus(event);
          setStatusText(textStatus);
          updateMessage(assistantId, (message) => ({
            ...message,
            phase: event?.event_type === "step_started" ? "tool" : message.phase,
            status: textStatus,
            taskEvents: event ? mergeTaskEvents(message.taskEvents, [event]) : message.taskEvents,
          }));
        },
        task_snapshot(payload) {
          applyTaskSnapshot(assistantId, payload);
        },
        done(payload) {
          applyTaskSnapshot(assistantId, payload);
          const task = payload.task;
          const finalAnswer = payload.final_answer || task?.final_answer || "";
          const failed = task?.status === "failed";
          updateMessage(assistantId, (message) => ({
            ...message,
            message_id: payload.message_id || message.message_id,
            content: finalAnswer || task?.error_message || "任务已完成，但没有生成最终回答。",
            phase: failed ? "error" : "done",
            status: "",
            task,
            taskSteps: payload.steps || message.taskSteps,
            taskEvents: payload.events || message.taskEvents,
          }));
          setStatusText("");
        },
        error(payload) {
          throw new Error(payload.detail || "任务流式响应失败。");
        },
      },
    );
  }

  function applyTaskSnapshot(assistantId: string, payload: TaskSnapshotPayload, status = "") {
    const task = payload.task;
    if (task?.thread_id) {
      setThreadId(task.thread_id);
      writeString(STORAGE.threadId, task.thread_id);
    }
    setMessages((items) =>
      items.map((message) => {
        const isCurrentAssistant = message.client_id === assistantId;
        const isSameTask = task?.task_id && message.task?.task_id === task.task_id;
        if (!isCurrentAssistant && isSameTask && message.task?.status === "waiting_user") {
          return {
            ...message,
            task: { ...message.task, status: "succeeded" },
            phase: "done",
            status: "已收到补充",
          };
        }
        if (!isCurrentAssistant) return message;
        return {
          ...message,
          task: task || message.task,
          taskSteps: payload.steps || message.taskSteps,
          taskEvents: payload.events || message.taskEvents,
          phase: task?.status === "succeeded" ? "done" : task?.status === "failed" ? "error" : message.phase,
          status: status || (task ? `任务${taskStatusLabel(task.status)}` : message.status),
        };
      }),
    );
  }

  function handleComposerKeydown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }

  async function loadKnowledgeBases() {
    if (knowledgeLoading) return;
    setKnowledgeLoading(true);
    try {
      const payload = await request<RecordResponse<{ knowledge_bases: KnowledgeBaseRecord[] }>>("/v1/knowledge-bases?limit=500");
      const bases = payload.data.knowledge_bases || [];
      setKnowledgeBases(bases);
      const nextSelected = selectedKbId && bases.some((item) => item.kb_id === selectedKbId) ? selectedKbId : bases[0]?.kb_id || "";
      setSelectedKbId(nextSelected);
      if (nextSelected) {
        await loadDocuments(nextSelected);
      } else {
        setDocuments([]);
      }
      setKnowledgeLoaded(true);
    } catch (caught) {
      setUploadStatus(`知识库同步失败：${errorMessage(caught)}`);
    } finally {
      setKnowledgeLoading(false);
    }
  }

  async function createKnowledgeBase(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const target = event.currentTarget;
    const form = new FormData(target);
    const name = String(form.get("name") || "").trim();
    if (!name || knowledgeLoading) return;
    setKnowledgeLoading(true);
    setUploadStatus("");
    try {
      const payload = await request<RecordResponse<KnowledgeBaseRecord>>("/v1/knowledge-bases", {
        method: "POST",
        body: JSON.stringify({ name, visibility: "private" }),
      });
      const kb = payload.data;
      setSelectedKbId(kb.kb_id);
      setKnowledgeBases((items) => [kb, ...items.filter((item) => item.kb_id !== kb.kb_id)]);
      await loadDocuments(kb.kb_id);
      setKnowledgeLoaded(true);
      setUploadStatus("知识库已创建。");
      target.reset();
    } catch (caught) {
      setUploadStatus(errorMessage(caught));
    } finally {
      setKnowledgeLoading(false);
    }
  }

  async function selectKnowledgeBase(kbId: string) {
    if (!kbId || kbId === selectedKbId) return;
    setSelectedKbId(kbId);
    setDocuments([]);
    setUploadStatus("");
    await loadDocuments(kbId);
  }

  async function loadDocuments(kbId: string) {
    const payload = await request<RecordResponse<{ documents: DocumentRecord[] }>>(`/v1/knowledge-bases/${encodeURIComponent(kbId)}/documents?limit=500`);
    setDocuments(payload.data.documents || []);
  }

  async function uploadKnowledgeFile(file: File | undefined) {
    if (!file || !selectedKbId || knowledgeLoading) return;
    const kbId = selectedKbId;
    setKnowledgeLoading(true);
    setUploadStatus(`正在上传 ${file.name}...`);
    try {
      const params = new URLSearchParams({ filename: file.name, parser_mode: parserMode });
      await binaryRequest<RecordResponse<unknown>>(apiBase, `/v1/knowledge-bases/${encodeURIComponent(kbId)}/documents?${params}`, {
        token,
        body: file,
        contentType: file.type || "application/pdf",
      });
      setUploadStatus("已加入后台解析队列。");
      await loadDocuments(kbId);
    } catch (caught) {
      setUploadStatus(errorMessage(caught));
    } finally {
      setKnowledgeLoading(false);
    }
  }

  async function deleteKnowledgeBase(kbId: string) {
    if (!kbId || knowledgeLoading) return;
    const kb = knowledgeBases.find((item) => item.kb_id === kbId);
    if (!window.confirm(`删除「${kb?.name || "这个知识库"}」？`)) return;
    const previousBases = knowledgeBases;
    const previousSelected = selectedKbId;
    const previousDocuments = documents;
    const nextBases = knowledgeBases.filter((item) => item.kb_id !== kbId);
    const nextSelected = previousSelected === kbId ? nextBases[0]?.kb_id || "" : previousSelected;
    setKnowledgeBases(nextBases);
    setSelectedKbId(nextSelected);
    if (previousSelected === kbId) setDocuments([]);
    setUploadStatus("正在后台删除知识库...");
    try {
      await request<RecordResponse<{ status: string }>>(`/v1/knowledge-bases/${encodeURIComponent(kbId)}`, { method: "DELETE" });
      if (previousSelected === kbId && nextSelected) {
        await loadDocuments(nextSelected);
      }
      setUploadStatus("知识库已删除。");
    } catch (caught) {
      setKnowledgeBases(previousBases);
      setSelectedKbId(previousSelected);
      setDocuments(previousDocuments);
      setUploadStatus(errorMessage(caught));
    }
  }

  async function deleteKnowledgeDocument(docId: string) {
    if (!docId || !selectedKbId || knowledgeLoading) return;
    const kbId = selectedKbId;
    const doc = documents.find((item) => item.doc_id === docId);
    if (!window.confirm(`删除「${doc?.file_name || doc?.title || "这个文档"}」？`)) return;
    const previousDocuments = documents;
    setDocuments((items) => items.filter((item) => item.doc_id !== docId));
    setUploadStatus("正在后台删除文档...");
    try {
      await request<RecordResponse<{ status: string }>>(
        `/v1/knowledge-bases/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}`,
        { method: "DELETE" },
      );
      setUploadStatus("文档已删除。");
    } catch (caught) {
      setDocuments(previousDocuments);
      setUploadStatus(errorMessage(caught));
    }
  }

  function scrollMessagesToBottom() {
    requestAnimationFrame(() => {
      if (messagesRef.current) {
        messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
      }
    });
  }

  if (!token) {
    return (
      <main className="grid min-h-dvh place-items-center overflow-y-auto px-4 py-4 sm:px-6 sm:py-6">
        <section className="grid w-full max-w-[980px] overflow-hidden rounded-3xl border border-violet-200/80 bg-white/82 shadow-2xl shadow-violet-200/30 backdrop-blur-xl md:min-h-[560px] md:grid-cols-[0.9fr_1fr]">
          <div className="flex flex-col justify-between gap-5 border-b border-violet-100 bg-gradient-to-br from-violet-50 via-white to-emerald-50 px-6 py-6 md:border-b-0 md:border-r md:px-8 md:py-8">
            <div>
              <div className="flex items-center gap-4 md:block">
                <KyuriAvatar size="lg" />
                <div className="md:mt-5">
                  <p className="text-xs font-black uppercase tracking-[0.28em] text-violet-700">Kyuriagents</p>
                  <h1 className="mt-1 text-3xl font-black text-stone-950 md:text-4xl">{authMode === "login" ? "欢迎回来" : "创建账号"}</h1>
                </div>
              </div>
              <p className="mt-5 max-w-sm text-sm leading-7 text-stone-600">
                你的 Agent 工作台已经准备好。登录后即可使用对话、知识库、长期记忆与任务模式。
              </p>
            </div>
            <p className="mt-auto rounded-2xl border border-violet-100 bg-white/70 px-4 py-3 text-xs font-semibold text-stone-600">
              作者联系方式：<span className="text-violet-700">{AUTHOR_EMAIL}</span>
            </p>
          </div>
          <div className="flex min-h-0 flex-col justify-center p-6 sm:p-7 md:p-8">
            <div className="mb-5 grid grid-cols-2 gap-2 rounded-xl bg-violet-50 p-1">
              <Button variant={authMode === "login" ? "default" : "ghost"} type="button" onClick={() => setAuthMode("login")}>
                登录
              </Button>
              <Button variant={authMode === "register" ? "default" : "ghost"} type="button" onClick={() => setAuthMode("register")}>
                注册
              </Button>
            </div>
            <form className="grid gap-3" onSubmit={handleAuthSubmit}>
              <label className="grid gap-1.5 text-xs font-bold text-stone-600">
                租户
                <Input name="tenant" defaultValue="default" autoComplete="organization" />
              </label>
              <label className="grid gap-1.5 text-xs font-bold text-stone-600">
                邮箱
                <Input name="email" type="email" autoComplete="email" required />
              </label>
              <label className="grid gap-1.5 text-xs font-bold text-stone-600">
                密码
                <Input name="password" type="password" autoComplete={authMode === "login" ? "current-password" : "new-password"} minLength={8} required />
              </label>
              {authMode === "register" ? (
                <label className="grid gap-1.5 text-xs font-bold text-stone-600">
                  昵称
                  <Input name="displayName" autoComplete="name" />
                </label>
              ) : null}
              <p className="min-h-5 text-xs text-rose-700">{error}</p>
              <Button type="submit" disabled={loading}>
                {loading ? "请稍候" : authMode === "login" ? "登录" : "注册并进入"}
              </Button>
            </form>
          </div>
        </section>
      </main>
    );
  }

  return (
    <div className="grid h-dvh grid-cols-1 overflow-hidden bg-stone-50/20 lg:grid-cols-[292px_minmax(0,1fr)]">
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-30 grid h-dvh w-[min(310px,88vw)] -translate-x-full grid-rows-[auto_auto_minmax(0,1fr)_auto] border-r border-stone-200/80 bg-white/80 shadow-2xl shadow-stone-300/30 backdrop-blur-xl transition lg:static lg:w-auto lg:translate-x-0 lg:shadow-none",
          sidebarOpen && "translate-x-0",
        )}
      >
        <div className="flex items-center justify-between gap-3 border-b border-stone-200/80 p-4">
          <div className="flex min-w-0 items-center gap-3">
            <KyuriAvatar size="sm" />
            <div className="min-w-0">
              <p className="truncate text-sm font-black text-stone-950">Kyuriagents</p>
              <p className="truncate text-xs text-stone-500">{email}</p>
            </div>
          </div>
          <Button size="icon" variant="secondary" type="button" onClick={startNewThread} title="新对话">
            <Plus className="h-4 w-4" />
          </Button>
        </div>
        <nav className="grid gap-2 p-3">
          <NavButton icon={MessageSquare} active={activeView === "chat"} title="对话" subtitle="问答与工具调用" onClick={startNewThread} />
          <NavButton icon={BookOpen} active={activeView === "knowledge"} title="知识库" subtitle="上传与索引" onClick={() => setActiveView("knowledge")} />
          <NavButton icon={Settings} active={activeView === "settings"} title="设置" subtitle="MCP 与偏好" onClick={() => setActiveView("settings")} />
        </nav>
        <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] overflow-hidden px-3 pb-3">
          <p className="px-1 py-2 text-xs font-black text-stone-500">最近对话</p>
          <div className="workspace-scrollbar min-h-0 overflow-y-auto pr-1">
            {threads.length ? (
              threads.map((thread) => (
                <div
                  key={thread.thread_id}
                  className={cn(
                    "group mb-2 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-lg border border-transparent p-2 transition hover:border-emerald-100 hover:bg-white/80 hover:shadow-sm",
                    thread.thread_id === threadId && "border-violet-200 bg-white shadow-sm",
                  )}
                >
                  <button type="button" onClick={() => void selectThread(thread.thread_id)} className="min-w-0 cursor-pointer rounded-md p-1 text-left active:scale-[0.99]">
                    <span className="block truncate font-semibold text-stone-900">{thread.title || "未命名对话"}</span>
                    <span className="text-xs text-stone-500">{displayDate(thread.updated_at || thread.created_at)}</span>
                  </button>
                  <Button
                    size="icon"
                    variant="ghost"
                    type="button"
                    className="h-8 w-8 opacity-0 transition group-hover:opacity-100 focus-visible:opacity-100"
                    title="删除对话"
                    onClick={(event) => {
                      event.stopPropagation();
                      void deleteThread(thread.thread_id);
                    }}
                  >
                    <Trash2 className="h-3.5 w-3.5 text-rose-700" />
                  </Button>
                </div>
              ))
            ) : (
              <p className="px-1 text-xs text-stone-500">暂无对话</p>
            )}
          </div>
        </div>
        <div className="flex items-center justify-between gap-3 border-t border-stone-200/80 p-4">
          <span className="truncate text-xs text-stone-500">{email}</span>
          <Button size="sm" variant="ghost" type="button" onClick={() => void logout()}>
            <LogOut className="h-4 w-4" />
            退出
          </Button>
        </div>
      </aside>
      <main className="min-h-0 min-w-0">
        {activeView === "knowledge" ? renderKnowledgeView() : activeView === "settings" ? renderSettingsView() : renderChatView()}
      </main>
    </div>
  );

  function renderChatView() {
    return (
      <section className="grid h-dvh grid-rows-[auto_minmax(0,1fr)_auto]">
        <header className="flex min-h-18 items-center justify-between gap-4 border-b border-stone-200/80 bg-white/65 px-4 py-3 backdrop-blur-xl lg:px-7">
          <div className="flex min-w-0 items-center gap-3">
            <Button className="lg:hidden" size="icon" variant="secondary" type="button" onClick={() => setSidebarOpen(true)}>
              <Menu className="h-4 w-4" />
            </Button>
            <div className="min-w-0">
              <p className="text-xs font-black text-violet-700">当前线程</p>
              <h1 className="truncate text-xl font-black text-stone-950">{currentThread?.title || "新对话"}</h1>
            </div>
          </div>
          <div className="flex items-center justify-end gap-3">
            <Switch checked={taskMode} disabled={loading} label="任务" onCheckedChange={setTaskMode} />
            <Switch checked={ragEnabled} disabled={loading} label="知识库" onCheckedChange={setRagEnabled} />
            <Switch checked={webSearchEnabled} disabled={loading} label="联网" onCheckedChange={setWebSearchEnabled} />
            <span className="hidden max-w-52 truncate rounded-full border border-violet-200 bg-white px-3 py-2 text-xs font-semibold text-stone-500 md:inline">
              {statusText || (loading ? "处理中" : "就绪")}
            </span>
            <span className="hidden items-center gap-2 rounded-full border border-violet-200 bg-white px-2 py-1.5 md:flex">
              <KyuriAvatar size="sm" className="h-7 w-7" />
              <span className="text-xs font-black">Kyuri</span>
            </span>
          </div>
        </header>
        <div ref={messagesRef} className="workspace-scrollbar min-h-0 overflow-auto px-4 py-6 lg:px-7">
          <div className="mx-auto grid w-full max-w-4xl gap-4">
            {messages.length ? (
              messages.map((message) => <MessageBubble key={message.message_id || message.client_id} message={message} />)
            ) : (
              <div className="grid min-h-[52dvh] place-items-center text-center">
                <div>
                  <Bot className="mx-auto mb-4 h-10 w-10 text-violet-500" />
                  <p className="text-xs font-black text-violet-700">新的对话</p>
                  <h2 className="mt-2 text-3xl font-black text-stone-950">今天要处理什么？</h2>
                </div>
              </div>
            )}
          </div>
        </div>
        <div className="bg-gradient-to-t from-stone-100 via-stone-100/95 to-transparent px-4 pb-5 pt-3 lg:px-7">
          <form className="mx-auto grid max-w-4xl grid-cols-[minmax(0,1fr)_auto] gap-3 rounded-xl border border-stone-200 bg-white p-2 shadow-xl shadow-stone-300/30" onSubmit={sendMessage}>
            <Textarea
              name="message"
              rows={2}
              disabled={loading}
              placeholder={
                waitingTask
                  ? "补充当前任务信息，Enter 发送，Shift + Enter 换行"
                  : "和 Kyuri 说点什么，Enter 发送，Shift + Enter 换行"
              }
              className="min-h-14 border-0 focus:ring-0"
              onKeyDown={handleComposerKeydown}
            />
            <Button type="submit" disabled={loading} className="self-end">
              <Send className="h-4 w-4" />
              发送
            </Button>
          </form>
          {error ? <p className="mx-auto mt-2 max-w-4xl text-xs text-rose-700">{error}</p> : null}
        </div>
      </section>
    );
  }

  function renderKnowledgeView() {
    return (
      <section className="workspace-scrollbar h-dvh overflow-auto">
        <header className="flex min-h-18 items-center justify-between gap-4 border-b border-stone-200/80 bg-white/65 px-4 py-3 backdrop-blur-xl lg:px-7">
          <div>
            <p className="text-xs font-black text-violet-700">知识库</p>
            <h1 className="text-xl font-black text-stone-950">{selectedKb?.name || "文档解析与索引"}</h1>
          </div>
          <form className="grid grid-cols-[minmax(160px,260px)_auto] gap-2" onSubmit={createKnowledgeBase}>
            <Input name="name" placeholder="新的知识库名称" required />
            <Button type="submit" variant="secondary" disabled={knowledgeLoading}>
              新建
            </Button>
          </form>
        </header>
        <div className="grid gap-5 p-4 lg:grid-cols-[minmax(0,1fr)_320px] lg:p-7">
          <section className="rounded-xl border border-stone-200 bg-white/75 p-4 shadow-sm">
            <div className="workspace-scrollbar mb-4 flex gap-2 overflow-x-auto pb-1">
              {knowledgeBases.length ? (
                knowledgeBases.map((kb) => (
                  <div
                    key={kb.kb_id}
                    className={cn(
                      "grid min-w-48 grid-rows-[1fr_auto] rounded-xl border bg-white",
                      kb.kb_id === selectedKbId ? "border-emerald-300 bg-emerald-50" : "border-stone-200",
                    )}
                  >
                    <button type="button" onClick={() => void selectKnowledgeBase(kb.kb_id)} className="grid gap-1 p-3 text-left">
                      <span className="truncate text-sm font-black text-stone-950">{kb.name || "未命名知识库"}</span>
                      <span className="text-xs text-stone-500">{kb.visibility || "private"}</span>
                    </button>
                    <Button className="m-2 justify-self-end" size="sm" variant="destructive" type="button" onClick={() => void deleteKnowledgeBase(kb.kb_id)}>
                      <Trash2 className="h-3.5 w-3.5" />
                      删除
                    </Button>
                  </div>
                ))
              ) : (
                <p className="text-sm text-stone-500">还没有知识库</p>
              )}
            </div>
            <label
              className={cn(
                "grid min-h-64 cursor-pointer place-items-center rounded-xl border border-dashed border-violet-300 bg-gradient-to-br from-violet-50 to-emerald-50 text-center",
                !selectedKbId && "cursor-not-allowed opacity-60",
              )}
            >
              <input
                className="hidden"
                type="file"
                accept="application/pdf,.pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,.docx,text/plain,.txt,.text"
                disabled={!selectedKbId || knowledgeLoading}
                onChange={(event) => void uploadKnowledgeFile(event.currentTarget.files?.[0])}
              />
              <span className="grid gap-3">
                <span className="mx-auto grid h-16 w-16 place-items-center rounded-xl bg-rose-100 font-black text-rose-500">DOC</span>
                <span className="text-2xl font-black text-stone-950">拖入 PDF / Word / TXT</span>
                <span className="text-sm text-stone-500">{selectedKbId ? "上传后进入后台队列" : "先创建或选择一个知识库"}</span>
                <span className="mx-auto inline-flex h-10 items-center rounded-lg bg-emerald-700 px-4 text-sm font-bold text-white">
                  <Upload className="mr-2 h-4 w-4" />
                  选择文件
                </span>
                {uploadStatus ? <span className="text-xs text-amber-700">{uploadStatus}</span> : null}
              </span>
            </label>
          </section>
          <section className="grid content-start gap-3 rounded-xl border border-stone-200 bg-white/75 p-4 shadow-sm">
            {["接收文件", "本地 / MCP 解析", "分块向量化", "写入 ES + Milvus"].map((item) => (
              <div key={item} className="flex min-h-12 items-center gap-3 rounded-lg border border-stone-200 bg-white px-3">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-700" />
                <strong className="text-sm">{item}</strong>
              </div>
            ))}
          </section>
        </div>
        <section className="mx-4 mb-7 rounded-xl border border-stone-200 bg-white/75 p-4 shadow-sm lg:mx-7">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-black text-stone-600">最近文档</h2>
            <span className="text-xs font-bold text-stone-500">后台处理，不阻塞聊天</span>
          </div>
          <div className="grid gap-2">
            {documents.length ? (
              documents.map((doc) => (
                <div key={doc.doc_id} className="flex min-h-14 items-center justify-between gap-4 rounded-lg border border-stone-200 bg-white px-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-black text-stone-950">{doc.file_name || doc.title || doc.doc_id}</p>
                    <p className="truncate text-xs text-stone-500">{doc.title || doc.file_name}</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <span
                      className={cn(
                        "rounded-full px-3 py-1 text-xs font-black",
                        doc.status === "failed"
                          ? "bg-rose-100 text-rose-700"
                          : doc.status === "succeeded"
                            ? "bg-emerald-100 text-emerald-800"
                            : "bg-stone-100 text-stone-600",
                      )}
                    >
                      {doc.status || "processing"}
                    </span>
                    <Button size="sm" variant="destructive" type="button" onClick={() => void deleteKnowledgeDocument(doc.doc_id)}>
                      删除
                    </Button>
                  </div>
                </div>
              ))
            ) : (
              <div className="grid min-h-36 place-items-center rounded-lg bg-white/60 text-center">
                <div>
                  <FileText className="mx-auto mb-2 h-8 w-8 text-stone-400" />
                  <p className="text-sm font-black">等待知识库</p>
                </div>
              </div>
            )}
          </div>
        </section>
      </section>
    );
  }

  function renderSettingsView() {
    return (
      <section className="workspace-scrollbar h-dvh overflow-auto">
        <header className="flex min-h-18 items-center justify-between gap-4 border-b border-stone-200/80 bg-white/65 px-4 py-3 backdrop-blur-xl lg:px-7">
          <div>
            <p className="text-xs font-black text-violet-700">设置</p>
            <h1 className="text-xl font-black text-stone-950">连接与偏好</h1>
          </div>
        </header>
        <div className="grid gap-5 p-4 lg:grid-cols-3 lg:p-7">
          <section className="rounded-xl border border-stone-200 bg-white/75 p-5 shadow-sm">
            <h2 className="mb-4 text-base font-black">账号</h2>
            <dl className="grid gap-3 text-sm">
              <div>
                <dt className="text-xs font-bold text-stone-500">邮箱</dt>
                <dd className="truncate font-semibold">{email}</dd>
              </div>
              <div>
                <dt className="text-xs font-bold text-stone-500">租户</dt>
                <dd className="truncate font-semibold">{user?.tenant_id || "default"}</dd>
              </div>
            </dl>
          </section>
          <section className="rounded-xl border border-stone-200 bg-white/75 p-5 shadow-sm">
            <h2 className="mb-4 text-base font-black">MCP 与解析</h2>
            <label className="grid gap-1.5 text-xs font-bold text-stone-600">
              解析模式
              <select
                value={parserMode}
                onChange={(event) => {
                  setParserMode(event.currentTarget.value as ParserMode);
                  setSettingsStatus("解析偏好已保存。");
                }}
                className="h-10 rounded-lg border border-stone-200 bg-white px-3 text-sm text-stone-950 outline-none focus:border-emerald-600/50 focus:ring-2 focus:ring-emerald-600/15"
              >
                <option value="auto">自动选择</option>
                <option value="local">本地解析</option>
                <option value="mcp">MCP 解析</option>
              </select>
            </label>
            <div className="mt-3 grid gap-1 rounded-lg border border-stone-200 bg-white/70 p-3 text-xs text-stone-500">
              <span>KYURIAGENTS_INGESTION_MCP_CONFIG_PATH</span>
              <span>KYURIAGENTS_INGESTION_MCP_TOOL_NAME</span>
            </div>
            {settingsStatus ? <p className="mt-3 text-xs text-emerald-800">{settingsStatus}</p> : null}
          </section>
          <section className="rounded-xl border border-stone-200 bg-white/75 p-5 shadow-sm">
            <h2 className="mb-4 text-base font-black">默认能力</h2>
            <div className="grid gap-4">
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-semibold">知识库检索</span>
                <Switch checked={ragEnabled} onCheckedChange={setRagEnabled} />
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="inline-flex items-center gap-2 text-sm font-semibold">
                  <Globe2 className="h-4 w-4 text-emerald-700" />
                  联网搜索
                </span>
                <Switch checked={webSearchEnabled} onCheckedChange={setWebSearchEnabled} />
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-semibold">任务模式</span>
                <Switch checked={taskMode} onCheckedChange={setTaskMode} />
              </div>
              <form
                className="grid gap-2"
                onSubmit={(event) => {
                  event.preventDefault();
                  const form = new FormData(event.currentTarget);
                  persistApiBase(String(form.get("apiBase") || ""));
                  setSettingsStatus("连接偏好已保存。");
                }}
              >
                <Input name="apiBase" defaultValue={apiBase} aria-label="API 地址" />
                <Button type="submit" variant="secondary">
                  保存
                </Button>
              </form>
            </div>
          </section>
        </div>
        <p className="mx-auto mt-16 max-w-3xl px-4 text-center text-xs font-semibold text-stone-500">
          如果您在使用时遇到了问题，请联系 <span className="text-violet-700">{AUTHOR_EMAIL}</span>，感谢！
        </p>
      </section>
    );
  }
}

function NavButton({
  icon: Icon,
  active,
  title,
  subtitle,
  onClick,
}: {
  icon: typeof MessageSquare;
  active: boolean;
  title: string;
  subtitle: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex min-h-16 cursor-pointer items-center gap-3 rounded-xl border border-transparent px-3 text-left transition-all hover:-translate-y-0.5 hover:bg-white/70 hover:shadow-sm active:translate-y-0 active:scale-[0.99]",
        active && "border-emerald-200 bg-emerald-50 shadow-[inset_4px_0_0_#047857]",
      )}
    >
      <Icon className="h-5 w-5 shrink-0 text-emerald-700" />
      <span className="min-w-0">
        <span className="block truncate text-sm font-black text-stone-950">{title}</span>
        <span className="block truncate text-xs text-stone-500">{subtitle}</span>
      </span>
    </button>
  );
}

function MessageBubble({ message }: { message: MessageRecord }) {
  const isAssistant = message.role === "assistant";
  return (
    <article className={cn("flex", isAssistant ? "justify-start" : "justify-end")}>
      <div
        className={cn(
          "max-w-[min(760px,92%)] rounded-xl border px-4 py-3 shadow-sm",
          isAssistant ? "border-violet-200 bg-white/95" : "border-emerald-200 bg-emerald-50",
        )}
      >
        <div className="mb-1 text-xs font-black text-stone-500">{isAssistant ? "Kyuri" : "你"}</div>
        <MessageText content={message.content} />
        {isAssistant && (message.task || message.taskSteps?.length || message.taskEvents?.length) ? <TaskPlanPanel message={message} /> : null}
        {message.status ? <div className="mt-2 text-xs font-semibold text-emerald-800">{message.status}</div> : null}
      </div>
    </article>
  );
}

function MessageText({ content }: { content: string }) {
  return (
    <div className="markdown-body break-words text-sm leading-7 text-stone-950">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1 className="mb-3 mt-4 text-xl font-black leading-snug text-stone-950">{children}</h1>,
          h2: ({ children }) => <h2 className="mb-2.5 mt-4 text-lg font-black leading-snug text-stone-950">{children}</h2>,
          h3: ({ children }) => <h3 className="mb-2 mt-3.5 text-base font-black leading-snug text-stone-950">{children}</h3>,
          h4: ({ children }) => <h4 className="mb-2 mt-3 text-sm font-black leading-snug text-stone-950">{children}</h4>,
          p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
          strong: ({ children }) => <strong className="font-black text-stone-950">{children}</strong>,
          ul: ({ children }) => <ul className="mb-3 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
          ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
          li: ({ children }) => <li className="pl-1">{children}</li>,
          hr: () => <hr className="my-4 border-stone-200" />,
          blockquote: ({ children }) => <blockquote className="my-3 border-l-4 border-violet-200 pl-3 text-stone-600">{children}</blockquote>,
          a: ({ children, href }) => (
            <a className="font-bold text-emerald-800 underline decoration-emerald-300 underline-offset-2" href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
          table: ({ children }) => <table className="my-3 w-full border-collapse overflow-hidden rounded-lg text-xs">{children}</table>,
          thead: ({ children }) => <thead className="bg-stone-100 text-stone-800">{children}</thead>,
          th: ({ children }) => <th className="border border-stone-200 px-2 py-1.5 text-left font-black">{children}</th>,
          td: ({ children }) => <td className="border border-stone-200 px-2 py-1.5 align-top">{children}</td>,
          code: ({ children }) => <code className="rounded bg-stone-100 px-1 py-0.5 text-[0.92em] font-semibold text-stone-900">{children}</code>,
          pre: ({ children }) => <pre className="my-3 overflow-auto rounded-lg bg-stone-950 p-3 text-xs leading-5 text-stone-50">{children}</pre>,
        }}
      >
        {content || ""}
      </ReactMarkdown>
    </div>
  );
}

function TaskPlanPanel({ message }: { message: MessageRecord }) {
  const steps = message.taskSteps || [];
  const events = message.taskEvents || [];
  const finished = ["succeeded", "failed", "cancelled"].includes(message.task?.status || "");
  const [open, setOpen] = useState(!finished);
  return (
    <section className="mt-3 rounded-lg border border-stone-200 bg-stone-50/80 p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div>
          <p className="text-xs font-black text-violet-700">任务计划</p>
          <p className="text-xs text-stone-500">{message.task?.task_id || "等待任务编号"}</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="rounded-full border border-stone-200 bg-white px-2.5 py-1 text-xs font-black text-stone-700">
            {taskStatusLabel(message.task?.status)}
          </span>
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="rounded-full border border-stone-200 bg-white px-2.5 py-1 text-xs font-black text-emerald-800 transition hover:border-emerald-200 hover:bg-emerald-50"
          >
            {open ? "收起" : "展开"}
          </button>
        </div>
      </div>
      {!open ? (
        <p className="text-xs text-stone-500">
          {steps.length ? `${steps.length} 个步骤，${events.length} 条事件。` : "任务详情已收起。"}
        </p>
      ) : (
        <>
          {steps.length ? (
            <div className="grid gap-2">
              {steps.map((step) => (
                <TaskStepItem key={step.step_id} step={step} />
              ))}
            </div>
          ) : (
            <p className="text-xs text-stone-500">正在等待 Planner 生成步骤。</p>
          )}
          {events.length ? (
            <div className="mt-3 border-t border-stone-200 pt-2">
              <p className="mb-1 text-xs font-black text-stone-500">最近事件</p>
              <div className="grid max-h-28 gap-1 overflow-auto pr-1">
                {events.slice(-5).map((event) => (
                  <p key={event.event_id} className="truncate text-[11px] text-stone-500">
                    {taskEventStatus(event)}
                  </p>
                ))}
              </div>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}

function TaskStepItem({ step }: { step: NonNullable<MessageRecord["taskSteps"]>[number] }) {
  const [open, setOpen] = useState(false);
  const hasOutput = Boolean(step.output?.trim());
  return (
    <div className="grid gap-1 rounded-lg border border-white bg-white/80 p-2">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <span className="block truncate text-xs font-black text-stone-900">
            {step.step_index + 1}. {step.title}
          </span>
          <span className="text-[11px] font-semibold text-violet-700">{stepKindLabel(step.kind)}</span>
        </div>
        <span className={cn("shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-black", stepStatusClass(step.status))}>
          {stepStatusLabel(step.status)}
        </span>
      </div>
      {step.tool_name ? <p className="truncate text-[11px] font-semibold text-emerald-700">工具：{step.tool_name}</p> : null}
      {step.input && Object.keys(step.input).length ? <p className="truncate text-[11px] text-stone-500">输入：{JSON.stringify(step.input)}</p> : null}
      {hasOutput ? (
        <div className="mt-1 rounded-md border border-stone-100 bg-stone-50 px-2 py-1.5">
          <div className="mb-1 flex items-center justify-between gap-2">
            <span className="text-[11px] font-black text-stone-500">输出结果</span>
            <button
              type="button"
              onClick={() => setOpen((value) => !value)}
              className="text-[11px] font-black text-emerald-800 hover:text-emerald-950"
            >
              {open ? "收起" : "展开"}
            </button>
          </div>
          <p className="whitespace-pre-wrap break-words text-[11px] leading-5 text-stone-700">
            {open ? step.output : stepOutputPreview(step.output)}
          </p>
        </div>
      ) : null}
      {step.error_message ? <p className="text-[11px] text-rose-700">{step.error_message}</p> : null}
    </div>
  );
}
