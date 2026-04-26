"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useTranslation } from "react-i18next";
import {
  ArrowLeft,
  Bot,
  Loader2,
  Paperclip,
  Send,
  X,
  Image as ImageIcon,
  Plus,
  Save,
  Download,
  History,
  Trash2,
  Pencil,
  Check,
  ChevronRight,
} from "lucide-react";
import dynamic from "next/dynamic";
import { apiUrl, wsUrl } from "@/lib/api";
import AssistantResponse from "@/components/common/AssistantResponse";
import { readFileAsDataUrl, extractBase64FromDataUrl } from "@/lib/file-attachments";

const SaveToNotebookModal = dynamic(
  () => import("@/components/notebook/SaveToNotebookModal"),
  { ssr: false },
);

// ── Types ──────────────────────────────────────────────────────

interface BotInfo {
  bot_id: string;
  name: string;
  running: boolean;
}

interface BotSession {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

interface ChatMsg {
  role: "user" | "assistant";
  content: string;
  thinking?: string[];
  attachments?: { type: string; filename: string; base64: string }[];
}

interface PendingAttachment {
  type: string;
  filename: string;
  base64: string;
  previewUrl?: string;
}

// ── Helpers ────────────────────────────────────────────────────

function timeGroupLabel(dateStr: string): string {
  const d = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  if (diffDays < 1) return "Today";
  if (diffDays < 2) return "Yesterday";
  if (diffDays < 7) return "Last 7 days";
  return "Earlier";
}

function formatTime(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

function groupByTime(sessions: BotSession[]): Record<string, BotSession[]> {
  const groups: Record<string, BotSession[]> = {};
  // Pre-defined order
  const order = ["Today", "Yesterday", "Last 7 days", "Earlier"];
  for (const s of sessions) {
    const label = timeGroupLabel(s.updated_at);
    if (!groups[label]) groups[label] = [];
    groups[label].push(s);
  }
  // Sort keys by order
  const sorted: Record<string, BotSession[]> = {};
  for (const k of order) {
    if (groups[k]) sorted[k] = groups[k];
  }
  return sorted;
}

// ── Component ──────────────────────────────────────────────────

export default function BotChatPage() {
  const { botId, sessionId: sessionIdParam } = useParams<{ botId: string; sessionId?: string[] }>();
  const router = useRouter();
  const { t } = useTranslation();

  // ── Core state ──
  const [bot, setBot] = useState<BotInfo | null>(null);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [thinking, setThinking] = useState<string[]>([]);
  const thinkingRef = useRef<string[]>([]);
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const [dragging, setDragging] = useState(false);
  const dragCounter = useRef(0);

  // ── Session management state ──
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(
    sessionIdParam?.[0] || null,
  );
  const [sessions, setSessions] = useState<BotSession[]>([]);
  const [showSessionList, setShowSessionList] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [draftTitle, setDraftTitle] = useState("");

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const titleInputRef = useRef<HTMLInputElement | null>(null);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    });
  }, []);

  // ── Session CRUD operations ──

  const loadSessions = useCallback(async () => {
    try {
      const res = await fetch(apiUrl(`/api/v1/tutorbot/${botId}/sessions`));
      if (res.ok) {
        const data: BotSession[] = await res.json();
        setSessions(data);
      }
    } catch {}
  }, [botId]);

  const createNewSession = useCallback(async () => {
    try {
      const res = await fetch(apiUrl(`/api/v1/tutorbot/${botId}/sessions`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (res.ok) {
        const { session_id } = await res.json();
        setCurrentSessionId(session_id);
        setMessages([]);
        router.push(`/agents/${botId}/chat/${session_id}`);
        await loadSessions();
      }
    } catch {}
  }, [botId, router, loadSessions]);

  const switchSession = useCallback((sid: string) => {
    setCurrentSessionId(sid);
    setShowSessionList(false);
    router.push(`/agents/${botId}/chat/${sid}`);
  }, [botId, router]);

  const renameSession = useCallback(async (sid: string, newTitle: string) => {
    try {
      const res = await fetch(apiUrl(`/api/v1/tutorbot/${botId}/sessions/${sid}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newTitle }),
      });
      if (res.ok) {
        setSessions((prev) =>
          prev.map((s) => (s.session_id === sid ? { ...s, title: newTitle } : s)),
        );
        if (currentSessionId === sid) {
          // Update displayed title if needed
        }
      }
    } catch {}
  }, [botId, currentSessionId]);

  const deleteSession = useCallback(async (sid: string) => {
    try {
      const res = await fetch(apiUrl(`/api/v1/tutorbot/${botId}/sessions/${sid}`), {
        method: "DELETE",
      });
      if (res.ok) {
        setSessions((prev) => prev.filter((s) => s.session_id !== sid));
        if (sid === currentSessionId) {
          // Create a new session after deleting current
          setCurrentSessionId(null);
          setMessages([]);
          router.push(`/agents/${botId}/chat`);
          createNewSession();
        }
      }
    } catch {}
  }, [botId, currentSessionId, router, createNewSession]);

  // ── Download Markdown ──

  const handleDownloadMarkdown = useCallback(() => {
    const firstUserMsg = messages.find((m) => m.role === "user");
    const title = firstUserMsg
      ? firstUserMsg.content.slice(0, 80).replace(/\n/g, " ")
      : bot?.name ?? botId ?? "Chat";

    const mdLines = [`# ${title}`, ""];
    for (const m of messages) {
      const label = m.role === "user" ? "Student" : "Teacher";
      mdLines.push(`## ${label}`, "", m.content, "", "---", "");
    }

    const blob = new Blob([mdLines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title.replace(/[^a-zA-Z0-9\u4e00-\u9fff]/g, "_")}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [messages, bot, botId]);

  // ── Title editing ──

  const startEditTitle = useCallback(() => {
    const current = sessions.find((s) => s.session_id === currentSessionId);
    setDraftTitle(current?.title ?? "New Chat");
    setEditingTitle(true);
    setTimeout(() => titleInputRef.current?.select(), 50);
  }, [sessions, currentSessionId]);

  const confirmEditTitle = useCallback(() => {
    const trimmed = draftTitle.trim();
    if (trimmed && currentSessionId) {
      renameSession(currentSessionId, trimmed);
    }
    setEditingTitle(false);
  }, [draftTitle, currentSessionId, renameSession]);

  // ── File handling ──

  const fileToAttachment = useCallback((f: File): Promise<PendingAttachment> => {
    return new Promise((resolve, reject) => {
      readFileAsDataUrl(f)
        .then((raw) => {
          const isImage = f.type.startsWith("image/");
          const b64 = extractBase64FromDataUrl(raw);
          resolve({
            type: isImage ? "image" : "file",
            filename: f.name,
            base64: b64,
            previewUrl: isImage ? raw : undefined,
          });
        })
        .catch(reject);
    });
  }, []);

  const handleAttachClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    const next = await Promise.all(files.map(fileToAttachment));
    setAttachments((prev) => [...prev, ...next]);
    e.target.value = "";
  }, [fileToAttachment]);

  const removeAttachment = useCallback((index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
  }, []);

  // ── Drag & Drop ──

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    if (e.dataTransfer.types.includes("Files")) setDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current -= 1;
    if (dragCounter.current === 0) setDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    dragCounter.current = 0;
    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    const next = await Promise.all(files.map(fileToAttachment));
    setAttachments((prev) => [...prev, ...next]);
  }, [fileToAttachment]);

  // ── Paste ──

  const handlePaste = useCallback(async (event: React.ClipboardEvent) => {
    const items = Array.from(event.clipboardData.items);
    const imageFiles = items
      .filter((item) => item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter((f): f is File => f !== null);
    if (!imageFiles.length) return;
    event.preventDefault();
    const next = await Promise.all(imageFiles.map(fileToAttachment));
    setAttachments((prev) => [...prev, ...next]);
  }, [fileToAttachment]);

  // ── Init effect: load bot info + sessions + history ──

  useEffect(() => {
    // Load bot info
    fetch(apiUrl(`/api/v1/tutorbot/${botId}`))
      .then((r) => (r.ok ? r.json() : null))
      .then(setBot)
      .catch(() => setBot(null));

    // Determine active session from URL
    const sid = sessionIdParam?.[0] || null;
    if (sid) {
      setCurrentSessionId(sid);
      // Load history for this session
      fetch(apiUrl(`/api/v1/tutorbot/${botId}/history?session_id=${sid}`))
        .then((r) => (r.ok ? r.json() : []))
        .then((history: { role: string; content: string | unknown[] }[]) => {
          const restored: ChatMsg[] = history
            .filter((m) => m.role === "user" || m.role === "assistant")
            .map((m) => {
              let content: string;
              if (typeof m.content === "string") {
                content = m.content;
              } else if (Array.isArray(m.content)) {
                content = m.content
                  .filter((c): c is { type: string; text?: string } =>
                    typeof c === "object" && c !== null && "type" in c)
                  .map((c) => (c.type === "text" ? c.text || "" : "[Image]"))
                  .join("\n");
                if (!content.trim()) content = "[Image]";
              } else {
                content = String(m.content ?? "");
              }
              return { role: m.role as "user" | "assistant", content };
            });
          if (restored.length) setMessages(restored);
        })
        .catch(() => {});
    } else {
      // No session ID — try to load legacy "default" session or newest session
      setCurrentSessionId(null);
      loadSessions().then(() => {
        // After sessions load, check if we should auto-load history
        // This is handled by the separate sessions-state effect below
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [botId]); // Only re-run when botId changes (not on every session switch)

  // Separate effect for session-specific history loading
  useEffect(() => {
    if (!sessionIdParam?.[0]) return;
    const sid = sessionIdParam[0];
    setCurrentSessionId(sid);
    fetch(apiUrl(`/api/v1/tutorbot/${botId}/history?session_id=${sid}`))
      .then((r) => (r.ok ? r.json() : []))
      .then((history: { role: string; content: string | unknown[] }[]) => {
        const restored: ChatMsg[] = history
          .filter((m) => m.role === "user" || m.role === "assistant")
          .map((m) => {
            let content: string;
            if (typeof m.content === "string") {
              content = m.content;
            } else if (Array.isArray(m.content)) {
              content = m.content
                .filter((c): c is { type: string; text?: string } =>
                  typeof c === "object" && c !== null && "type" in c)
                .map((c) => (c.type === "text" ? c.text || "" : "[Image]"))
                .join("\n");
              if (!content.trim()) content = "[Image]";
            } else {
              content = String(m.content ?? "");
            }
            return { role: m.role as "user" | "assistant", content };
          });
        if (restored.length) setMessages(restored);
      })
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionIdParam?.[0]]); // Re-run when session param changes

  // Also load sessions list whenever we need it
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Auto-select "default" legacy session (or most recent) when no session is active
  useEffect(() => {
    if (currentSessionId || sessions.length === 0) return; // Already have a session or nothing to pick

    // Priority: "default" (legacy) > most recent by updated_at
    const legacy = sessions.find((s) => s.session_id === "default");
    const target = legacy || sessions[0];
    if (!target) return;

    setCurrentSessionId(target.session_id);
    // Load history for auto-selected session
    fetch(apiUrl(`/api/v1/tutorbot/${botId}/history?session_id=${target.session_id}`))
      .then((r) => (r.ok ? r.json() : []))
      .then((history: { role: string; content: string | unknown[] }[]) => {
        const restored: ChatMsg[] = history
          .filter((m) => m.role === "user" || m.role === "assistant")
          .map((m) => {
            let content: string;
            if (typeof m.content === "string") {
              content = m.content;
            } else if (Array.isArray(m.content)) {
              content = m.content
                .filter((c): c is { type: string; text?: string } =>
                  typeof c === "object" && c !== null && "type" in c)
                .map((c) => (c.type === "text" ? c.text || "" : "[Image]"))
                .join("\n");
              if (!content.trim()) content = "[Image]";
            } else {
              content = String(m.content ?? "");
            }
            return { role: m.role as "user" | "assistant", content };
          });
        if (restored.length) setMessages(restored);
      })
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessions, currentSessionId, botId]);

  // ── WebSocket ──

  useEffect(() => {
    const sid = currentSessionId || undefined;
    const wsUrlStr = wsUrl(`/api/v1/tutorbot/${botId}/ws${sid ? `?session_id=${sid}` : ""}`);
    const ws = new WebSocket(wsUrlStr);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === "thinking") {
        thinkingRef.current = [...thinkingRef.current, data.content];
        setThinking(thinkingRef.current);
        scrollToBottom();
      } else if (data.type === "content") {
        const snap = thinkingRef.current;
        setMessages((msgs) => [
          ...msgs,
          { role: "assistant", content: data.content, thinking: snap.length ? [...snap] : undefined },
        ]);
        thinkingRef.current = [];
        setThinking([]);
        scrollToBottom();
        // Refresh session stats after receiving response
        if (currentSessionId) {
          loadSessions();
        }
      } else if (data.type === "done") {
        setStreaming(false);
        setTimeout(() => inputRef.current?.focus(), 50);
      } else if (data.type === "proactive") {
        setMessages((msgs) => [...msgs, { role: "assistant", content: data.content }]);
        scrollToBottom();
      } else if (data.type === "error") {
        setMessages((msgs) => [...msgs, { role: "assistant", content: `Error: ${data.content}` }]);
        thinkingRef.current = [];
        setThinking([]);
        setStreaming(false);
      }
    };

    ws.onclose = () => {
      setStreaming(false);
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [botId, currentSessionId, scrollToBottom, loadSessions]);

  // ── Send ──

  const send = useCallback(() => {
    const text = input.trim();
    if ((!text && !attachments.length) || streaming || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const msgAttachments = attachments.map((a) => ({ type: a.type, filename: a.filename, base64: a.base64 }));

    setMessages((msgs) => [
      ...msgs,
      { role: "user", content: text, attachments: msgAttachments.length > 0 ? msgAttachments : undefined },
    ]);
    setInput("");
    setAttachments([]);
    setStreaming(true);
    setThinking([]);

    const payload: Record<string, unknown> = { content: text };
    if (msgAttachments.length > 0) {
      payload.attachments = msgAttachments;
    }
    wsRef.current.send(JSON.stringify(payload));
    scrollToBottom();
  }, [input, attachments, streaming, scrollToBottom]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    },
    [send],
  );

  // ── Derived state ──

  const currentSessionTitle = sessions.find((s) => s.session_id === currentSessionId)?.title ?? (bot?.name ?? botId);
  const groupedSessions = groupByTime(sessions.filter((s) => s.session_id !== currentSessionId));

  // ── Render ──

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-[var(--border)] px-5 py-3">
        <button
          onClick={() => router.push("/agents")}
          className="rounded-lg p-1.5 text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
        >
          <ArrowLeft className="h-4 w-4" />
        </button>
        <Bot className="h-4 w-4 text-[var(--muted-foreground)]" />

        {/* Session title (editable) */}
        {editingTitle ? (
          <div className="flex items-center gap-1.5 flex-1 max-w-[200px]">
            <input
              ref={titleInputRef}
              value={draftTitle}
              onChange={(e) => setDraftTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") confirmEditTitle();
                if (e.key === "Escape") setEditingTitle(false);
              }}
              onBlur={confirmEditTitle}
              className="flex-1 rounded-md border border-[var(--ring)] bg-transparent px-2 py-1 text-[13px] text-[var(--foreground)] outline-none"
              autoFocus
            />
            <button onClick={confirmEditTitle} className="p-1 text-emerald-500 hover:bg-emerald-500/10 rounded">
              <Check size={14} />
            </button>
          </div>
        ) : (
          <button
            onClick={startEditTitle}
            className="text-[14px] font-medium text-[var(--foreground)] hover:text-[var(--primary)] transition-colors truncate max-w-[240px]"
            title={currentSessionTitle}
          >
            {currentSessionTitle}
          </button>
        )}

        {bot?.running && (
          <span className="h-2 w-2 rounded-full bg-emerald-500" />
        )}

        {/* Action buttons */}
        <div className="ml-auto flex items-center gap-1">
          <button
            onClick={() => setShowSaveModal(true)}
            className="rounded-lg p-1.5 text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
            title={t("Save to notebook")}
          >
            <Save className="h-4 w-4" />
          </button>
          <button
            onClick={handleDownloadMarkdown}
            disabled={messages.length === 0}
            className="rounded-lg p-1.5 text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)] disabled:opacity-30"
            title={t("Download Markdown")}
          >
            <Download className="h-4 w-4" />
          </button>
          <button
            onClick={createNewSession}
            className="rounded-lg p-1.5 text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
            title={t("New conversation")}
          >
            <Plus className="h-4 w-4" />
          </button>
          <button
            onClick={() => setShowSessionList(!showSessionList)}
            className={`rounded-lg p-1.5 transition-colors ${
              showSessionList
                ? "bg-[var(--muted)] text-[var(--foreground)]"
                : "text-[var(--muted-foreground)] hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
            }`}
            title={t("Session history")}
          >
            <History className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Session List Panel */}
      {showSessionList && (
        <div className="border-b border-[var(--border)] bg-[var(--muted)]/20 max-h-[300px] overflow-y-auto">
          <div className="px-5 py-2 text-[11px] font-medium uppercase tracking-wider text-[var(--muted-foreground)]">
            {t("Session History")}
          </div>
          {sessions.length === 0 ? (
            <div className="px-5 py-6 text-center text-[13px] text-[var(--muted-foreground)]">
              {t("No conversations yet.")}
            </div>
          ) : (
            <div className="pb-2">
              {/* Current session (if exists) */}
              {currentSessionId && (() => {
                const cs = sessions.find((s) => s.session_id === currentSessionId);
                if (!cs) return null;
                return (
                  <div className="mx-3 mb-1 flex items-center gap-2 rounded-lg bg-[var(--primary)]/10 px-3 py-2">
                    <ChevronRight className="h-3.5 w-3.5 shrink-0 text-[var(--primary)]" />
                    <span className="flex-1 truncate text-[13px] font-medium text-[var(--foreground)]">{cs.title}</span>
                    <span className="text-[11px] text-[var(--muted-foreground)]">{formatTime(cs.updated_at)}</span>
                  </div>
                );
              })()}

              {/* Grouped other sessions */}
              {Object.entries(groupedSessions).map(([groupLabel, groupSessions]) => (
                <div key={groupLabel}>
                  <div className="mt-2 px-5 pb-1 text-[11px] font-medium uppercase tracking-wider text-[var(--muted-foreground)]">
                    {groupLabel}
                  </div>
                  {groupSessions.map((s) => (
                    <SessionItem
                      key={s.session_id}
                      session={s}
                      isActive={false}
                      onSelect={() => switchSession(s.session_id)}
                      onRename={(newTitle) => renameSession(s.session_id, newTitle)}
                      onDelete={() => deleteSession(s.session_id)}
                    />
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-6 [scrollbar-gutter:stable]">
        <div className="mx-auto max-w-[720px] space-y-5">
          {messages.length === 0 && !streaming && (
            <div className="flex flex-col items-center justify-center pt-24 text-center">
              <div className="mb-3 rounded-xl bg-[var(--muted)] p-3 text-[var(--muted-foreground)]">
                <Bot size={22} />
              </div>
              <p className="text-[14px] font-medium text-[var(--foreground)]">
                {t("Chat with {{name}}", { name: bot?.name ?? botId })}
              </p>
              <p className="mt-1 text-[13px] text-[var(--muted-foreground)]">
                {t("Send a message to start the conversation.")}
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={msg.role === "user" ? "flex justify-end" : ""}>
              {msg.role === "user" ? (
                <div className="max-w-[80%] space-y-2">
                  {msg.attachments?.map((att, j) =>
                    att.type === "image" ? (
                      <img
                        key={j}
                        src={`data:${att.filename.endsWith(".png") ? "image/png" : "image/jpeg"};base64,${att.base64}`}
                        alt={att.filename}
                        className="max-w-[280px] rounded-xl"
                      />
                    ) : null,
                  )}
                  {msg.content && (
                    <div className="max-w-[80%] rounded-2xl rounded-br-md bg-[var(--primary)] px-4 py-2.5 text-[14px] text-[var(--primary-foreground)]">
                      {msg.content}
                    </div>
                  )}
                </div>
              ) : (
                <div className="max-w-full">
                  {msg.thinking && msg.thinking.length > 0 && (
                    <details className="mb-2">
                      <summary className="cursor-pointer text-[12px] text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
                        {t("Thinking ({{count}} steps)", { count: msg.thinking.length })}
                      </summary>
                      <div className="mt-1 space-y-1 border-l-2 border-[var(--border)] pl-3">
                        {msg.thinking.map((th, j) => (
                          <p key={j} className="text-[12px] text-[var(--muted-foreground)]">{th}</p>
                        ))}
                      </div>
                    </details>
                  )}
                  <AssistantResponse content={msg.content} />
                </div>
              )}
            </div>
          ))}

          {/* Streaming indicator */}
          {streaming && (
            <div className="space-y-2">
              {thinking.length > 0 && (
                <div className="space-y-1 border-l-2 border-[var(--border)] pl-3">
                  {thinking.map((th, i) => (
                    <p key={i} className="text-[12px] text-[var(--muted-foreground)]">{th}</p>
                  ))}
                </div>
              )}
              <div className="flex items-center gap-2 text-[13px] text-[var(--muted-foreground)]">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>{thinking.length > 0 ? t("Working...") : t("Thinking...")}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Attachment previews */}
      {attachments.length > 0 && (
        <div className="border-t border-[var(--border)]/40 px-5 pt-2">
          <div className="mx-auto flex max-w-[720px] flex-wrap gap-2">
            {attachments.map((att, i) =>
              att.previewUrl ? (
                <div key={i} className="relative group">
                  <img
                    src={att.previewUrl}
                    alt={att.filename}
                    className="h-14 w-14 rounded-lg object-cover border border-[var(--border)]/40"
                  />
                  <button
                    onClick={() => removeAttachment(i)}
                    className="absolute -top-1.5 -right-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-white opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X size={10} strokeWidth={2.5} />
                  </button>
                </div>
              ) : (
                <span
                  key={i}
                  className="inline-flex items-center gap-1 rounded-lg border border-[var(--border)]/40 bg-[var(--muted)]/30 px-2 py-1 text-[11px] text-[var(--muted-foreground)]"
                >
                  <ImageIcon size={11} /> {att.filename}
                  <button onClick={() => removeAttachment(i)} className="ml-0.5 opacity-60 hover:opacity-100">
                    <X size={9} />
                  </button>
                </span>
              ),
            )}
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-[var(--border)] px-5 py-3">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={handleFileSelect}
        />

        <div
          className={`relative mx-auto flex max-w-[720px] items-end gap-2 rounded-xl transition-colors ${
            dragging ? "bg-[var(--muted)]/30 ring-2 ring-[var(--primary)]/30" : ""
          }`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          {dragging && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center rounded-xl bg-[var(--muted)]/60 backdrop-blur-sm border-2 border-dashed border-[var(--primary)]/40">
              <Paperclip size={24} strokeWidth={1.5} className="text-[var(--primary)] mb-1" />
              <span className="text-[12px] font-medium text-[var(--primary)]">{t("Drop images here")}</span>
            </div>
          )}

          <button
            type="button"
            onClick={handleAttachClick}
            className="flex h-[42px] w-[42px] shrink-0 items-center justify-center rounded-xl text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)]/50 hover:text-[var(--foreground)]"
            aria-label={t("Attach file")}
            title={t("Upload image")}
          >
            <Paperclip size={18} strokeWidth={1.8} />
          </button>

          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={t("Type a message...")}
            rows={1}
            disabled={streaming}
            className="flex-1 resize-none rounded-xl border border-[var(--border)] bg-transparent px-4 py-2.5 text-[14px] text-[var(--foreground)] outline-none transition-colors focus:border-[var(--ring)] disabled:opacity-50 placeholder:text-[var(--muted-foreground)]/40"
          />
          <button
            onClick={send}
            disabled={streaming || (!input.trim() && !attachments.length)}
            className="flex h-[42px] w-[42px] items-center justify-center rounded-xl bg-[var(--primary)] text-[var(--primary-foreground)] transition-opacity hover:opacity-90 disabled:opacity-30"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Save to Notebook Modal */}
      <SaveToNotebookModal
        open={showSaveModal}
        payload={{
          recordType: "tutorbot",
          title: currentSessionTitle,
          userQuery: messages.find((m) => m.role === "user")?.content ?? "",
          output: messages.filter((m) => m.role === "assistant").map((m) => m.content).join("\n\n"),
          metadata: { source: "tutorbot", bot_id: botId, session_id: currentSessionId ?? undefined },
        }}
        messages={messages.map((m) => ({ role: m.role, content: m.content }))}
        onClose={() => setShowSaveModal(false)}
      />
    </div>
  );
}

// ── Session Item Sub-component ──────────────────────────────────

function SessionItem({
  session,
  isActive,
  onSelect,
  onRename,
  onDelete,
}: {
  session: BotSession;
  isActive: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(session.title);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (renaming) {
      setDraft(session.title);
      setTimeout(() => inputRef.current?.select(), 50);
    }
  }, [renaming, session.title]);

  const confirmRename = () => {
    const trimmed = draft.trim();
    if (trimmed) onRename(trimmed);
    setRenaming(false);
  };

  return (
    <div
      className={`mx-3 flex items-center gap-2 rounded-lg px-3 py-2 cursor-pointer transition-colors group ${
        isActive ? "bg-[var(--primary)]/10" : "hover:bg-[var(--muted)]"
      }`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={renaming ? undefined : onSelect}
    >
      {renaming ? (
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") confirmRename();
            if (e.key === "Escape") setRenaming(false);
          }}
          onBlur={confirmRename}
          onClick={(e) => e.stopPropagation()}
          className="flex-1 rounded border border-[var(--ring)] bg-transparent px-2 py-0.5 text-[13px] outline-none"
          autoFocus
        />
      ) : (
        <>
          <span className="flex-1 truncate text-[13px] text-[var(--foreground)]">{session.title}</span>
          <span className="text-[11px] text-[var(--muted-foreground)] shrink-0">{formatTime(session.updated_at)}</span>

          {/* Hover actions */}
          {hovered && (
            <div className="flex items-center gap-0.5 shrink-0" onClick={(e) => e.stopPropagation()}>
              <button
                onClick={() => setRenaming(true)}
                className="p-1 rounded text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--muted)]"
              >
                <Pencil size={12} />
              </button>
              <button
                onClick={onDelete}
                className="p-1 rounded text-red-400 hover:text-red-500 hover:bg-red-500/10"
              >
                <Trash2 size={12} />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
