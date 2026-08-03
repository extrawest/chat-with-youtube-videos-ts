import { Supadata } from "@supadata/js";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

export function extractYoutubeVideoId(url: string): string | null {
  const match = url.match(
    /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|shorts\/|watch\?v=|\&v=)([^#\&\?]*).*/,
  );
  return match && match[2].length === 11 ? match[2] : null;
}

export async function getYoutubeTranscriptDocs(
  videoId: string,
): Promise<Document[]> {
  try {
    const supadata = new Supadata({ apiKey: process.env.SUPADATA_API_KEY! });
    const result = await supadata.youtube.transcript({ videoId });

    if (!result?.content)
      throw new Error("No transcript available for this video.");

    let fullText = "";
    if (typeof result.content === "string") {
      fullText = result.content;
    } else if (Array.isArray(result.content)) {
      fullText = result.content
        .map((item: any) => (typeof item === "string" ? item : item.text || ""))
        .filter(Boolean)
        .join(" ");
    }

    if (!fullText || !fullText.trim()) {
      throw new Error("Transcript content is empty.");
    }

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2000,
      chunkOverlap: 200,
    });

    return await splitter.createDocuments([fullText], [{ videoId }]);
  } catch (error: any) {
    throw new Error(`Failed to fetch transcript: ${error.message || error}`);
  }
}
