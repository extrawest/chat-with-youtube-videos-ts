import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { extractYoutubeVideoId, getYoutubeTranscriptDocs } from "./youtube";
import {
  ingestTranscriptToVectorStore,
  checkVideoIngested,
} from "./vectorstore";
import { Document } from "@langchain/core/documents";

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-3.7-flash",
  apiKey: process.env.GEMINI_API_KEY,
});

export const IngestGraphState = Annotation.Root({
  url: Annotation<string>(),
  videoId: Annotation<string>(),
  fullText: Annotation<string>(),
  summary: Annotation<string>(),
  docs: Annotation<Document[]>({ reducer: (_, y) => y, default: () => [] }),
  chunksCount: Annotation<number>({ reducer: (_, y) => y, default: () => 0 }),
  status: Annotation<string>({ reducer: (_, y) => y, default: () => "idle" }),
  error: Annotation<string | null>({
    reducer: (_, y) => y,
    default: () => null,
  }),
});

async function extractVideoIdNode(state: typeof IngestGraphState.State) {
  if (!state.url || typeof state.url !== "string") {
    throw new Error("A valid YouTube URL string is required.");
  }

  const videoId = extractYoutubeVideoId(state.url);
  if (!videoId) {
    throw new Error("Invalid YouTube video URL.");
  }

  return { videoId, status: "extracted_id" };
}

async function fetchTranscriptNode(state: typeof IngestGraphState.State) {
  const alreadyIngested = await checkVideoIngested(state.videoId);
  if (alreadyIngested) {
    return { docs: [], fullText: "", status: "already_ingested" };
  }

  const { docs, fullText } = await getYoutubeTranscriptDocs(state.videoId);
  if (!docs || docs.length === 0) {
    throw new Error("No transcript documents parsed for this video.");
  }

  return { docs, fullText, status: "fetched_transcript" };
}

async function generateSummaryNode(state: typeof IngestGraphState.State) {
  if (state.status === "already_ingested" || !state.fullText) {
    return { summary: "", status: "skipped_summary" };
  }

  const prompt = `Provide a concise 150-word overview summary of this YouTube video transcript. Cover the main topic, key takeaways, and speaker context:

Transcript:
${state.fullText.substring(0, 8000)}`;

  try {
    const res = await llm.invoke([{ role: "user", content: prompt }]);
    const summary =
      typeof res.content === "string"
        ? res.content
        : JSON.stringify(res.content);
    return { summary, status: "generated_summary" };
  } catch {
    return { summary: "", status: "failed_summary" };
  }
}

async function ingestVectorstoreNode(state: typeof IngestGraphState.State) {
  if (state.status === "already_ingested") {
    return { chunksCount: 0, status: "completed" };
  }

  await ingestTranscriptToVectorStore(state.videoId, state.docs, state.summary);
  return {
    chunksCount: state.docs.length,
    status: "completed",
  };
}

const ingestWorkflow = new StateGraph(IngestGraphState)
  .addNode("extract_video_id", extractVideoIdNode, {
    retryPolicy: { maxAttempts: 2, backoffFactor: 1.5 },
  })
  .addNode("fetch_transcript", fetchTranscriptNode, {
    retryPolicy: { maxAttempts: 3, backoffFactor: 2 },
  })
  .addNode("generate_summary", generateSummaryNode, {
    retryPolicy: { maxAttempts: 2, backoffFactor: 1.5 },
  })
  .addNode("ingest_vectorstore", ingestVectorstoreNode, {
    retryPolicy: { maxAttempts: 3, backoffFactor: 2 },
  })
  .addEdge(START, "extract_video_id")
  .addEdge("extract_video_id", "fetch_transcript")
  .addEdge("fetch_transcript", "generate_summary")
  .addEdge("generate_summary", "ingest_vectorstore")
  .addEdge("ingest_vectorstore", END);

export const ingestGraph = ingestWorkflow.compile();
