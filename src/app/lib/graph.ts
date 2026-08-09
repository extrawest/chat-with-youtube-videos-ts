import {
  Annotation,
  StateGraph,
  START,
  END,
  MemorySaver,
} from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { tavily } from "@tavily/core";
import { BaseMessage, AIMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { similaritySearchWithScore } from "./vectorstore";

export const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  videoId: Annotation<string>(),
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

export function extractVideoIdFromThreadId(
  threadId?: string,
): string | undefined {
  if (!threadId) return undefined;
  const match = threadId.match(/^thread_([a-zA-Z0-9_-]{11})_\d+$/);
  return match ? match[1] : undefined;
}

async function vectorSearchNode(
  state: typeof GraphState.State,
  config?: { configurable?: { thread_id?: string } },
) {
  const lastMessage = state.messages[state.messages.length - 1];
  if (!lastMessage) {
    return { maxSimilarity: 0, contextDocs: [] };
  }

  const query =
    typeof lastMessage.content === "string"
      ? lastMessage.content
      : JSON.stringify(lastMessage.content);

  const effectiveVideoId =
    state.videoId ||
    extractVideoIdFromThreadId(config?.configurable?.thread_id);

  const searchResults = await similaritySearchWithScore(
    query,
    effectiveVideoId,
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
    videoId: effectiveVideoId || state.videoId,
    maxSimilarity: maxScore,
    contextDocs: docs,
  };
}

function routeByRelevance(
  state: typeof GraphState.State,
): "rag_generator" | "tavily_fallback" {
  const similarityThreshold = 0.75;
  return state.maxSimilarity >= similarityThreshold
    ? "rag_generator"
    : "tavily_fallback";
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

const workflow = new StateGraph(GraphState)
  .addNode("vector_search", vectorSearchNode, {
    retryPolicy: { maxAttempts: 3, backoffFactor: 2 },
  })
  .addNode("rag_generator", ragGeneratorNode, {
    retryPolicy: { maxAttempts: 3, backoffFactor: 2 },
  })
  .addNode("tavily_fallback", tavilyFallbackNode, {
    retryPolicy: { maxAttempts: 3, backoffFactor: 2 },
  })
  .addEdge(START, "vector_search")
  .addConditionalEdges("vector_search", routeByRelevance, {
    rag_generator: "rag_generator",
    tavily_fallback: "tavily_fallback",
  })
  .addEdge("rag_generator", END)
  .addEdge("tavily_fallback", END);

export const checkpointer = new MemorySaver();

export const graph = workflow.compile({
  checkpointer,
});
