import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import {
  extractYoutubeVideoId,
  getYoutubeTranscriptDocs,
} from "./youtube";
import { ingestTranscriptToVectorStore } from "./vectorstore";
import { Document } from "@langchain/core/documents";

export const IngestGraphState = Annotation.Root({
  url: Annotation<string>(),
  videoId: Annotation<string>(),
  docs: Annotation<Document[]>({
    reducer: (_, y) => y,
    default: () => [],
  }),
  chunksCount: Annotation<number>({
    reducer: (_, y) => y,
    default: () => 0,
  }),
  status: Annotation<string>({
    reducer: (_, y) => y,
    default: () => "idle",
  }),
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
  const docs = await getYoutubeTranscriptDocs(state.videoId);
  if (!docs || docs.length === 0) {
    throw new Error("No transcript documents parsed for this video.");
  }

  return { docs, status: "fetched_transcript" };
}

async function ingestVectorstoreNode(state: typeof IngestGraphState.State) {
  await ingestTranscriptToVectorStore(state.videoId, state.docs);
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
  .addNode("ingest_vectorstore", ingestVectorstoreNode, {
    retryPolicy: { maxAttempts: 3, backoffFactor: 2 },
  })
  .addEdge(START, "extract_video_id")
  .addEdge("extract_video_id", "fetch_transcript")
  .addEdge("fetch_transcript", "ingest_vectorstore")
  .addEdge("ingest_vectorstore", END);

export const ingestGraph = ingestWorkflow.compile();
