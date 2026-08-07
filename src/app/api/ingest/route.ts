import { NextRequest, NextResponse } from "next/server";
import { ingestGraph } from "@/app/lib/ingestGraph";
import { graph } from "@/app/lib/graph";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { url } = body || {};

    if (!url || typeof url !== "string" || !url.trim()) {
      return NextResponse.json(
        { error: "A valid YouTube URL string is required." },
        { status: 400 },
      );
    }

    const ingestResult = await ingestGraph.invoke({ url: url.trim() });

    if (ingestResult.error) {
      return NextResponse.json(
        { error: ingestResult.error },
        { status: 400 },
      );
    }

    const videoId = ingestResult.videoId;
    const threadId = `thread_${videoId}_${Date.now()}`;

    await graph.updateState(
      { configurable: { thread_id: threadId } },
      { videoId },
    );

    return NextResponse.json({
      success: true,
      videoId,
      threadId,
      chunksCount: ingestResult.chunksCount,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "Failed to process YouTube video transcript." },
      { status: 500 },
    );
  }
}
