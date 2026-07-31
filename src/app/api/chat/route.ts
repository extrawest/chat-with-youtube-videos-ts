import { NextRequest, NextResponse } from "next/server";
import { HumanMessage } from "@langchain/core/messages";
import { graph } from "@/app/lib/graph";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const { message, threadId } = await req.json();

    if (!message || !threadId) {
      return NextResponse.json(
        { error: "Both 'message' and 'threadId' parameters are required." },
        { status: 400 },
      );
    }

    const inputState = { messages: [new HumanMessage(message)] };
    const eventStream = graph.streamEvents(inputState, {
      configurable: { thread_id: threadId },
      version: "v2",
    });

    const encoder = new TextEncoder();

    const stream = new ReadableStream({
      async start(controller) {
        try {
          for await (const event of eventStream) {
            if (
              event.event === "on_chat_model_stream" &&
              event.data?.chunk?.content
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
