import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { Pool } from "pg";

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl:
    process.env.NODE_ENV === "production"
      ? { rejectUnauthorized: false }
      : false,
});

export const checkpointer = new PostgresSaver(pool);

let isSetup = false;

export async function ensureCheckpointerSetup(): Promise<void> {
  if (!isSetup && process.env.DATABASE_URL) {
    try {
      await checkpointer.setup();
      isSetup = true;
    } catch (err: any) {
      console.warn(`[PostgresSaver Setup Warning]: ${err.message || err}`);
    }
  }
}
