import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { tavily } from "@tavily/core";
import { BaseMessage, AIMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  similaritySearchWithScore,
  fetchVideoSummaryFromStore,
} from "./vectorstore";
import { checkpointer } from "./checkpointer";

export type UserIntent = "overview" | "specific_detail" | "off_topic";

export const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  videoId: Annotation<string>(),
  intent: Annotation<UserIntent>({
    reducer: (_, y) => y,
    default: () => "specific_detail",
  }),
  videoSummary: Annotation<string>(),
  maxSimilarity: Annotation<number>({
    reducer: (_, y) => y,
    default: () => 0,
  }),
  contextDocs: Annotation<string[]>({
    reducer: (_, y) => y,
    default: () => [],
  }),
});

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-3.6-flash",
  apiKey: process.env.GEMINI_API_KEY,
  streaming: true,
});

const googleAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");
const classifierModel = googleAI.getGenerativeModel({
  model: "gemini-3.6-flash",
});

const tavilyClient = tavily({
  apiKey: process.env.TAVILY_API_KEY,
});

export const tavilySearchTool = tool(
  async ({ query }: { query: string }) => {
    const res = await tavilyClient.search(query, {
      includeAnswer: true,
      maxResults: 3,
    });
    return JSON.stringify(res);
  },
  {
    name: "tavily_web_search",
    description:
      "Performs real-time web searches to answer queries when video transcript context is insufficient.",
    schema: z.object({
      query: z.string().describe("Search query string"),
    }),
  },
);

async function intentClassifierNode(state: typeof GraphState.State) {
  const lastMessage = state.messages[state.messages.length - 1];
  const query =
    typeof lastMessage.content === "string"
      ? lastMessage.content
      : JSON.stringify(lastMessage.content);

  let summary = state.videoSummary;
  if (!summary && state.videoId) {
    summary = (await fetchVideoSummaryFromStore(state.videoId)) || "";
  }

  const prompt = `Classify the user's question regarding a YouTube video.
Video Summary: "${summary.substring(0, 300)}"

Categories:
1. "overview": Question asks for a summary, topic overview, main points, or who is speaking.
2. "specific_detail": Question asks about specific facts, details, quotes, or numbers in the video.
3. "off_topic": Question is completely unrelated to the video topic.

Return ONLY ONE word (overview, specific_detail, or off_topic).

Question: ${query}`;

  try {
    const result = await classifierModel.generateContent(prompt);
    const ans = result.response.text().trim().toLowerCase();

    let intent: UserIntent = "specific_detail";
    if (ans.includes("overview")) intent = "overview";
    else if (ans.includes("off_topic")) intent = "off_topic";

    console.log(
      `[Intent Classifier]: Identified "${intent}" for question: "${query.substring(0, 60)}"`,
    );
    return { videoId: state.videoId, videoSummary: summary, intent };
  } catch (err: any) {
    console.warn(
      `[Intent Classifier Warning]: ${err.message || err}. Defaulting to specific_detail.`,
    );
    return {
      videoId: state.videoId,
      videoSummary: summary,
      intent: "specific_detail" as UserIntent,
    };
  }
}

async function vectorSearchNode(state: typeof GraphState.State) {
  const lastMessage = state.messages[state.messages.length - 1];
  const rawQuery =
    typeof lastMessage.content === "string"
      ? lastMessage.content
      : JSON.stringify(lastMessage.content);

  const searchResults = await similaritySearchWithScore(
    rawQuery,
    state.videoId,
    4,
  );

  if (!searchResults?.length) {
    return { maxSimilarity: 0, contextDocs: [] };
  }

  let maxScore = 0;
  const docs: string[] = [];

  for (const [doc, score] of searchResults) {
    if (score > maxScore) maxScore = score;
    docs.push(doc.pageContent);
  }

  return {
    maxSimilarity: maxScore,
    contextDocs: docs,
  };
}

async function videoSummaryNode(state: typeof GraphState.State) {
  const prompt = `You are an AI assistant analyzing a YouTube video.
Answer the user's question using the high-level video summary below.

Video Summary:
${state.videoSummary || "No summary available."}`;

  const response = await llm.invoke([
    { role: "system", content: prompt },
    ...state.messages,
  ]);

  return { messages: [response] };
}

async function ragGeneratorNode(state: typeof GraphState.State) {
  const context = state.contextDocs.join("\n\n---\n\n");

  const systemPrompt = `You are an AI assistant analyzing a YouTube video. 
Use ONLY the transcript snippets below to answer the user's question accurately.

Transcript Context:
${context}`;

  const response = await llm.invoke([
    { role: "system", content: systemPrompt },
    ...state.messages,
  ]);

  return { messages: [response] };
}

async function tavilyFallbackNode(state: typeof GraphState.State) {
  const lastMessage = state.messages[state.messages.length - 1];
  const query =
    typeof lastMessage.content === "string"
      ? lastMessage.content
      : JSON.stringify(lastMessage.content);

  const rawToolResult = await tavilySearchTool.invoke({ query });
  const tavilyResponse = JSON.parse(rawToolResult);

  const directAnswer =
    tavilyResponse.answer || "No direct answer found from web search.";
  const sourcesMarkdown = (tavilyResponse.results || [])
    .slice(0, 3)
    .map((r: any, i: number) => `${i + 1}. [${r.title}](${r.url})`)
    .join("\n");

  const formattedResponse = `*Notice: Low video relevance (${(
    state.maxSimilarity * 100
  ).toFixed(
    1,
  )}%). Falling back to Web Search.*\n\n${directAnswer}\n\n**Sources:**\n${sourcesMarkdown}`;

  return { messages: [new AIMessage(formattedResponse)] };
}

function routeByIntent(
  state: typeof GraphState.State,
): "video_summary" | "vector_search" | "tavily_fallback" {
  if (state.intent === "overview") return "video_summary";
  if (state.intent === "off_topic") return "tavily_fallback";
  return "vector_search";
}

function routeByRelevance(
  state: typeof GraphState.State,
): "rag_generator" | "tavily_fallback" {
  const similarityThreshold = 0.75;
  return state.maxSimilarity >= similarityThreshold
    ? "rag_generator"
    : "tavily_fallback";
}

const workflow = new StateGraph(GraphState)
  .addNode("intent_classifier", intentClassifierNode)
  .addNode("vector_search", vectorSearchNode, {
    retryPolicy: { maxAttempts: 3, backoffFactor: 2 },
  })
  .addNode("video_summary", videoSummaryNode, {
    retryPolicy: { maxAttempts: 2, backoffFactor: 1.5 },
  })
  .addNode("rag_generator", ragGeneratorNode, {
    retryPolicy: { maxAttempts: 3, backoffFactor: 2 },
  })
  .addNode("tavily_fallback", tavilyFallbackNode, {
    retryPolicy: { maxAttempts: 3, backoffFactor: 2 },
  })
  .addEdge(START, "intent_classifier")
  .addConditionalEdges("intent_classifier", routeByIntent, {
    video_summary: "video_summary",
    vector_search: "vector_search",
    tavily_fallback: "tavily_fallback",
  })
  .addConditionalEdges("vector_search", routeByRelevance, {
    rag_generator: "rag_generator",
    tavily_fallback: "tavily_fallback",
  })
  .addEdge("video_summary", END)
  .addEdge("rag_generator", END)
  .addEdge("tavily_fallback", END);

export const graph = workflow.compile({
  checkpointer,
});
