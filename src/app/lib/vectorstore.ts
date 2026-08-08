import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";
import { Document } from "@langchain/core/documents";

export function getDocumentEmbeddings(): GoogleGenerativeAIEmbeddings {
  return new GoogleGenerativeAIEmbeddings({
    model: "gemini-embedding-001",
    apiKey: process.env.GEMINI_API_KEY,
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    maxConcurrency: 10,
  });
}

export function getQueryEmbeddings(): GoogleGenerativeAIEmbeddings {
  return new GoogleGenerativeAIEmbeddings({
    model: "gemini-embedding-001",
    apiKey: process.env.GEMINI_API_KEY,
    taskType: TaskType.RETRIEVAL_QUERY,
    maxConcurrency: 10,
  });
}

export function getPineconeIndex() {
  const apiKey =
    process.env.PINECONE_API_KEY?.trim().replace(/^["']|["']$/g, "") || "";
  const indexName = process.env.PINECONE_INDEX_NAME || "youtube-transcripts";
  const pinecone = new Pinecone({ apiKey });
  return pinecone.Index(indexName);
}

export async function checkVideoIngested(videoId: string): Promise<boolean> {
  try {
    const embeddings = getQueryEmbeddings();
    const probeVector = await embeddings.embedQuery("test probe");
    const index = getPineconeIndex();

    const checkRes = await index.query({
      vector: probeVector,
      topK: 1,
      filter: { videoId: { $eq: videoId } },
    });

    return Boolean(checkRes.matches && checkRes.matches.length > 0);
  } catch (err: any) {
    console.warn(`[Pinecone Check Warning]: ${err.message || err}`);
    return false;
  }
}

export async function ingestTranscriptToVectorStore(
  videoId: string,
  docs: Document[],
) {
  if (!docs?.length) return;

  const alreadyIngested = await checkVideoIngested(videoId);
  if (alreadyIngested) {
    console.log(`Video ${videoId} transcript chunks already ingested.`);
    return;
  }

  const docsWithMetadata = docs
    .filter((doc) => doc.pageContent && doc.pageContent.trim().length > 0)
    .map((doc, idx) => ({
      id: `${videoId}_chunk_${idx}`,
      metadata: { videoId, pageContent: doc.pageContent },
    }));

  if (!docsWithMetadata.length) return;

  const embeddings = getDocumentEmbeddings();
  const vecs = await embeddings.embedDocuments(
    docsWithMetadata.map((d) => d.metadata.pageContent),
  );

  if (!vecs?.length || !vecs[0]?.length) {
    throw new Error(
      "Embedding generation returned empty vectors. Check GEMINI_API_KEY.",
    );
  }

  const records = docsWithMetadata.map((doc, i) => ({
    id: doc.id,
    values: vecs[i],
    metadata: doc.metadata,
  }));

  const index = getPineconeIndex();
  const BATCH_SIZE = 100;

  for (let i = 0; i < records.length; i += BATCH_SIZE) {
    const batch = records.slice(i, i + BATCH_SIZE);
    await index.upsert({ records: batch });
  }

  console.log(
    `Ingested ${records.length} chunks into Pinecone in batches of ${BATCH_SIZE} for video ${videoId}.`,
  );
}

export async function similaritySearchWithScore(
  query: string,
  filterVideoId?: string,
  k = 4,
): Promise<[Document, number][]> {
  const embeddings = getQueryEmbeddings();
  const queryVec = await embeddings.embedQuery(query);

  const index = getPineconeIndex();
  const res = await index.query({
    vector: queryVec,
    topK: k,
    includeMetadata: true,
    filter: filterVideoId ? { videoId: { $eq: filterVideoId } } : undefined,
  });

  return (res.matches || []).map((m) => [
    new Document({
      pageContent: (m.metadata?.pageContent as string) || "",
      metadata: m.metadata || {},
    }),
    m.score || 0,
  ]);
}
