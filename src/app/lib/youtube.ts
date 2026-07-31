import { YoutubeTranscript } from "youtube-transcript";
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
    const rawTranscript = await YoutubeTranscript.fetchTranscript(videoId);
    if (!rawTranscript?.length) {
      throw new Error("No transcript available for this video.");
    }

    const fullText = rawTranscript.map((t) => t.text).join(" ");
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2000,
      chunkOverlap: 200,
    });

    return await splitter.createDocuments([fullText], [{ videoId }]);
  } catch (error: any) {
    throw new Error(`Failed to fetch transcript: ${error.message || error}`);
  }
}
