import { Supadata } from "@supadata/js";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

export function extractYoutubeVideoId(url: string): string | null {
  const match = url.match(
    /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|shorts\/|watch\?v=|\&v=)([^#\&\?]*).*/,
  );
  return match && match[2].length === 11 ? match[2] : null;
}

export function cleanTranscriptText(rawText: string): string {
  return rawText
    .replace(/\[.*?\]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export async function getYoutubeTranscriptDocs(
  videoId: string,
): Promise<{ docs: Document[]; fullText: string }> {
  try {
    const supadata = new Supadata({ apiKey: process.env.SUPADATA_API_KEY! });
    const result = await supadata.youtube.transcript({ videoId });

    if (!result?.content)
      throw new Error("No transcript available for this video.");

    let rawText = "";
    if (typeof result.content === "string") {
      rawText = result.content;
    } else if (Array.isArray(result.content)) {
      rawText = result.content
        .map((item: any) => (typeof item === "string" ? item : item.text || ""))
        .filter(Boolean)
        .join(" ");
    }

    const fullText = cleanTranscriptText(rawText);
    if (!fullText)
      throw new Error("Transcript content is empty after cleaning.");

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 600,
      chunkOverlap: 150,
    });

    const docs = await splitter.createDocuments([fullText], [{ videoId }]);
    return { docs, fullText };
  } catch (error: any) {
    throw new Error(`Failed to fetch transcript: ${error.message || error}`);
  }
}
