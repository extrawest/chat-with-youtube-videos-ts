import { NextRequest, NextResponse } from "next/server";
import { HumanMessage } from "@langchain/core/messages";
import { graph, extractVideoIdFromThreadId } from "@/app/lib/graph";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const threadId = searchParams.get("threadId");

    if (!threadId) {
      return NextResponse.json(
        { error: "Query parameter 'threadId' is required." },
        { status: 400 },
      );
    }

    const state = await graph.getState({
      configurable: { thread_id: threadId },
    });

    const videoId = state.values?.videoId || extractVideoIdFromThreadId(threadId);

    return NextResponse.json({
      threadId,
      videoId,
      values: {
        ...state.values,
        videoId,
      },
      next: state.next,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "Failed to retrieve graph state." },
      { status: 500 },
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const { message, threadId } = await req.json();

    if (!message || typeof message !== "string" || !message.trim()) {
      return NextResponse.json(
        { error: "A valid non-empty 'message' string is required." },
        { status: 400 },
      );
    }

    if (!threadId || typeof threadId !== "string" || !threadId.trim()) {
      return NextResponse.json(
        { error: "A valid 'threadId' string is required." },
        { status: 400 },
      );
    }

    const videoId = extractVideoIdFromThreadId(threadId);
    const inputState = {
      messages: [new HumanMessage(message)],
      ...(videoId ? { videoId } : {}),
    };

    const eventStream = graph.streamEvents(inputState, {
      configurable: { thread_id: threadId },
      version: "v2",
    });

    const encoder = new TextEncoder();

    const stream = new ReadableStream({
      async start(controller) {
        try {
          for await (const event of eventStream) {
            const activeNode = event.metadata?.langgraph_node;
            if (
              event.event === "on_chat_model_stream" &&
              event.data?.chunk?.content &&
              (activeNode === "rag_generator" || activeNode === "video_summary")
            ) {
              controller.enqueue(encoder.encode(event.data.chunk.content));
            } else if (
              event.event === "on_chain_end" &&
              event.name === "tavily_fallback" &&
              event.data?.output?.messages
            ) {
              const lastMsg = event.data.output.messages[0]?.content;
              if (lastMsg) controller.enqueue(encoder.encode(lastMsg));
            }
          }
        } catch (err: any) {
          controller.enqueue(
            encoder.encode(`\n\n[Error during generation: ${err.message}]`),
          );
        } finally {
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "Failed to process chat request." },
      { status: 500 },
    );
  }
}
