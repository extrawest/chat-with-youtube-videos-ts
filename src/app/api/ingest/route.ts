import { NextRequest, NextResponse } from "next/server";
import {
  extractYoutubeVideoId,
  getYoutubeTranscriptDocs,
} from "@/app/lib/youtube";
import { ingestTranscriptToVectorStore } from "@/app/lib/vectorstore";
import { graph } from "@/app/lib/graph";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const { url } = await req.json();

    if (!url || typeof url !== "string") {
      return NextResponse.json(
        { error: "A valid YouTube URL string is required." },
        { status: 400 },
      );
    }

    const videoId = extractYoutubeVideoId(url);
    if (!videoId) {
      return NextResponse.json(
        { error: "Invalid YouTube video URL." },
        { status: 400 },
      );
    }

    const docs = await getYoutubeTranscriptDocs(videoId);
    await ingestTranscriptToVectorStore(videoId, docs);

    const threadId = `thread_${videoId}_${Date.now()}`;
    await graph.updateState(
      { configurable: { thread_id: threadId } },
      { videoId },
    );

    return NextResponse.json({
      success: true,
      videoId,
      threadId,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "Failed to process YouTube video transcript." },
      { status: 500 },
    );
  }
}
