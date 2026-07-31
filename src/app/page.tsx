"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function Home() {
  const [url, setUrl] = useState<string>("https://www.youtube.com/watch?v=U9mJuUkhUzk");
  const [isIngesting, setIsIngesting] = useState<boolean>(false);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  const handleIngest = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!url.trim()) return;

    setIsIngesting(true);
    setIngestError(null);
    setMessages([]);

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to process video.");

      setVideoId(data.videoId);
      setThreadId(data.threadId);
    } catch (err: any) {
      setIngestError(err.message || "An unexpected error occurred.");
    } finally {
      setIsIngesting(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !threadId || isStreaming) return;

    const text = inputMessage;
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInputMessage("");
    setIsStreaming(true);

    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, threadId }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || "Failed to generate answer");
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error("No readable stream received");

      let accumulatedAnswer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        accumulatedAnswer += decoder.decode(value, { stream: true });

        setMessages((prev) => {
          const newMsgs = [...prev];
          newMsgs[newMsgs.length - 1] = {
            role: "assistant",
            content: accumulatedAnswer,
          };
          return newMsgs;
        });
      }
    } catch (err: any) {
      setMessages((prev) => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1] = {
          role: "assistant",
          content: `Error: ${err.message || "Something went wrong"}`,
        };
        return newMsgs;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-[#f4f4f5] flex flex-col items-center justify-center p-6 font-sans">
      <main className="w-full max-w-5xl flex flex-col items-center justify-center gap-6">
        <div className="w-full bg-[#18181b] border border-[#27272a] rounded-lg p-4">
          <form onSubmit={handleIngest} className="flex gap-3">
            <input
              type="text"
              placeholder="Paste YouTube Video URL..."
              className="flex-1 bg-[#09090b] border border-[#27272a] rounded px-3 py-2.5 text-sm text-[#f4f4f5] focus:outline-none focus:border-[#a1a1aa]"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={isIngesting}
            />
            <button
              type="submit"
              className="bg-[#f4f4f5] text-[#09090b] font-medium text-sm px-5 py-2.5 rounded hover:bg-[#e4e4e7] disabled:opacity-50"
              disabled={isIngesting || !url.trim()}
            >
              {isIngesting ? "Loading..." : "Load Video"}
            </button>
          </form>

          {ingestError && (
            <p className="text-xs text-red-400 mt-2">{ingestError}</p>
          )}
        </div>

        {videoId && (
          <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
            <div className="bg-[#18181b] border border-[#27272a] rounded-lg overflow-hidden h-130 w-full flex items-center justify-center">
              <iframe
                src={`https://www.youtube.com/embed/${videoId}`}
                title="YouTube video player"
                className="w-full h-full border-0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>

            <div className="bg-[#18181b] border border-[#27272a] rounded-lg flex flex-col h-130 w-full">
              <div className="flex-1 p-4 overflow-y-auto space-y-4">
                {messages.length === 0 ? null : (
                  messages.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}
                    >
                      <div
                        className={`max-w-[85%] text-sm rounded p-3 ${msg.role === "user"
                          ? "bg-[#27272a] text-[#f4f4f5] whitespace-pre-wrap"
                          : "bg-[#09090b] border border-[#27272a] text-[#f4f4f5]"
                          }`}
                      >
                        {msg.role === "user" ? (
                          msg.content
                        ) : (
                          <ReactMarkdown
                            components={{
                              a: ({ ...props }) => (
                                <a
                                  {...props}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="underline text-blue-400 hover:text-blue-300"
                                />
                              ),
                              p: ({ ...props }) => (
                                <p {...props} className="mb-2 last:mb-0 leading-relaxed" />
                              ),
                              ul: ({ ...props }) => (
                                <ul {...props} className="list-disc list-inside mb-2 space-y-1" />
                              ),
                              ol: ({ ...props }) => (
                                <ol {...props} className="list-decimal list-inside mb-2 space-y-1" />
                              ),
                              li: ({ ...props }) => (
                                <li {...props} className="leading-relaxed" />
                              ),
                              strong: ({ ...props }) => (
                                <strong {...props} className="font-semibold text-white" />
                              ),
                            }}
                          >
                            {msg.content || "..."}
                          </ReactMarkdown>
                        )}
                      </div>
                    </div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>

              <div className="p-3 border-t border-[#27272a] bg-[#18181b]">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleSendMessage();
                  }}
                  className="flex gap-2"
                >
                  <input
                    type="text"
                    placeholder="Ask a question..."
                    className="flex-1 bg-[#09090b] border border-[#27272a] rounded px-3 py-2 text-sm text-[#f4f4f5] focus:outline-none focus:border-[#a1a1aa]"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    disabled={!threadId || isStreaming}
                  />
                  <button
                    type="submit"
                    className="bg-[#f4f4f5] text-[#09090b] font-medium text-sm px-4 py-2 rounded hover:bg-[#e4e4e7] disabled:opacity-50"
                    disabled={!threadId || !inputMessage.trim() || isStreaming}
                  >
                    Send
                  </button>
                </form>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
