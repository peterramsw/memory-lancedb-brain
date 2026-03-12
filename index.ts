/**
 * memory-lancedb-brain - OpenClaw intelligent memory plugin
 * Phase 1 skeleton only: config loading + LanceDB connect + embedder init
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { connectDb, ensureTables } from "./src/storage.js";
import { createEmbedder, type EmbeddingConfig } from "./src/embedding.js";

interface PluginConfig {
  embedding?: {
    apiKey?: string;
    model?: string;
    baseURL?: string;
    dimensions?: number;
  };
  dbPath?: string;
  owners?: Array<{
    owner_id: string;
    owner_namespace: string;
    channels?: string[];
  }>;
  agentWhitelist?: string[];
  retrieval?: {
    mode?: string;
    vectorWeight?: number;
    bm25Weight?: number;
    minScore?: number;
    rerank?: boolean;
    rerankApiKey?: string;
    rerankModel?: string;
    rerankEndpoint?: string;
    rerankProvider?: string;
    candidatePoolSize?: number;
    hardMinScore?: number;
  };
  distillation?: {
    model?: string;
    baseURL?: string;
    apiKey?: string;
  };
}

const DEFAULT_EMBEDDING: Required<EmbeddingConfig> = {
  apiKey: "local",
  model: "vllm/Forturne/Qwen3-Embedding-4B-NVFP4",
  baseURL: "http://127.0.0.1:32080/v1",
  dimensions: 2560,
};

export default function register(api: OpenClawPluginApi): void {
  void (async () => {
    try {
      const config = (api.config ?? {}) as PluginConfig;
      const dbPath = config.dbPath ?? ".openclaw-memory-lancedb-brain";
      const embeddingConfig: EmbeddingConfig = {
        ...DEFAULT_EMBEDDING,
        ...(config.embedding ?? {}),
      };

      const db = await connectDb(dbPath);
      await ensureTables(db);
      const embedder = createEmbedder(embeddingConfig);

      (api as any).__memoryLanceDbBrain = {
        db,
        embedder,
        config,
      };

      api.log.info("memory-lancedb-brain: Phase 1 initialized");
    } catch (error) {
      api.log.error(`memory-lancedb-brain: Phase 1 initialization failed: ${String(error)}`);
    }
  })();
}
