"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useTranslation } from "react-i18next";
import { ArrowLeft, Bot, Loader2, Paperclip, Send, X, Image as ImageIcon } from "lucide-react";
import { apiUrl, wsUrl } from "@/lib/api";
import AssistantResponse from "@/components/common/AssistantResponse";
import { readFileAsDataUrl, extractBase64FromDataUrl } from "@/lib/file-attachments";

interface BotInfo {
  bot_id: string;
  name: string;
  running: boolean;
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

export default function BotChatPage() {
  const { botId } = useParams<{ botId: string }>();
  const router = useRouter();
  const { t } = useTranslation();

  const [bot, setBot] = useState<BotInfo | null>(null);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [thinking, setThinking] = useState<string[]>([]);
  const thinkingRef = useRef<string[]>([]);
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const [dragging, setDragging] = useState(false);
  const dragCounter = useRef(0);

  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    });
  }, []);

  // --- File handling ---
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

  // --- Drag & Drop ---
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

  // --- Paste ---
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

  // --- Bot & History loading ---
  useEffect(() => {
    fetch(apiUrl(`/api/v1/tutorbot/${botId}`))
      .then((r) => (r.ok ? r.json() : null))
      .then(setBot)
      .catch(() => setBot(null));

    fetch(apiUrl(`/api/v1/tutorbot/${botId}/history`))
      .then((r) => (r.ok ? r.json() : []))
      .then((history: { role: string; content: string | unknown[] }[]) => {
        const restored: ChatMsg[] = history
          .filter((m) => m.role === "user" || m.role === "assistant")
          .map((m) => {
            // Handle multimodal content (array of {type, text/image_url} objects)
            // that can appear when image attachments were sent via TutorBot WS
            let content: string;
            if (typeof m.content === "string") {
              content = m.content;
            } else if (Array.isArray(m.content)) {
              // Extract text parts from multimodal array, skip image_url entries
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
  }, [botId]);

  // --- WebSocket ---
  useEffect(() => {
    const ws = new WebSocket(wsUrl(`/api/v1/tutorbot/${botId}/ws`));
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
  }, [botId, scrollToBottom]);

  // --- Send ---
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

    // Send text + attachments as JSON
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
        <span className="text-[14px] font-medium text-[var(--foreground)]">
          {bot?.name ?? botId}
        </span>
        {bot?.running && (
          <span className="h-2 w-2 rounded-full bg-emerald-500" />
        )}
      </div>

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
                  {/* Show user's attached images */}
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

      {/* Input area with drag-drop zone */}
      <div className="border-t border-[var(--border)] px-5 py-3">
        {/* Hidden file input */}
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
          {/* Drag overlay */}
          {dragging && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center rounded-xl bg-[var(--muted)]/60 backdrop-blur-sm border-2 border-dashed border-[var(--primary)]/40">
              <Paperclip size={24} strokeWidth={1.5} className="text-[var(--primary)] mb-1" />
              <span className="text-[12px] font-medium text-[var(--primary)]">{t("Drop images here")}</span>
            </div>
          )}

          {/* Upload button */}
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
    </div>
  );
}
