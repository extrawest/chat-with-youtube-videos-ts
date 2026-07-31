import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { Document } from "@langchain/core/documents";

export function getEmbeddings(): GoogleGenerativeAIEmbeddings {
  return new GoogleGenerativeAIEmbeddings({
    model: "gemini-embedding-001",
    apiKey: process.env.GEMINI_API_KEY,
    maxConcurrency: 10,
  });
}

function getPineconeIndex() {
  const apiKey =
    process.env.PINECONE_API_KEY?.trim().replace(/^["']|["']$/g, "") || "";
  const indexName = process.env.PINECONE_INDEX_NAME || "youtube-transcripts";
  const pinecone = new Pinecone({ apiKey });
  return pinecone.Index(indexName);
}

export async function ingestTranscriptToVectorStore(
  videoId: string,
  docs: Document[],
) {
  if (!docs?.length) return;

  const index = getPineconeIndex();

  try {
    const checkRes = await index.query({
      vector: new Array(3072).fill(0.01),
      topK: 1,
      filter: { videoId: { $eq: videoId } },
    });
    if (checkRes.matches && checkRes.matches.length > 0) return;
  } catch (err) {}

  const docsWithMetadata = docs
    .filter((doc) => doc.pageContent && doc.pageContent.trim().length > 0)
    .map((doc, idx) => ({
      id: `${videoId}_chunk_${idx}`,
      metadata: { videoId, pageContent: doc.pageContent },
    }));

  if (!docsWithMetadata.length) return;

  const embeddings = getEmbeddings();
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

  await index.upsert({ records });
}

export async function similaritySearchWithScore(
  query: string,
  filterVideoId?: string,
  k = 4,
): Promise<[Document, number][]> {
  const embeddings = getEmbeddings();
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
