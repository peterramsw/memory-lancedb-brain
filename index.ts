/**
 * memory-lancedb-brain - OpenClaw intelligent memory plugin
 * Phase 2: initialization + tool registration
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { MemoryStorage } from "./src/storage.js";
import { createEmbedder, type EmbeddingConfig } from "./src/embedding.js";
import { normalizeOwners } from "./src/owners.js";
import { registerAllMemoryTools } from "./src/tools.js";

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
    channels?: Record<string, string>;
  }>;
  agentWhitelist?: string[];
  retrieval?: {
    mode?: "hybrid" | "vector" | "keyword";
    vectorWeight?: number;
    bm25Weight?: number;
    minScore?: number;
    hardMinScore?: number;
    rerank?: boolean;
    rerankApiKey?: string;
    rerankModel?: string;
    rerankEndpoint?: string;
    candidatePoolSize?: number;
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

const DEFAULT_WHITELIST = [
  "main",
  "gb10-deploy",
  "peter-365",
  "plaw-coding-team",
  "tiffany-ops",
  "gb10-openclaw-pr-reviewer",
];

export default function register(api: OpenClawPluginApi & { logger?: any; pluginConfig?: unknown }): void {
  void (async () => {
    try {
      const config = ((api.pluginConfig ?? api.config ?? {}) as PluginConfig) ?? {};
      const dbPath = api.resolvePath?.(config.dbPath ?? "./data/memory-lancedb-brain") ?? (config.dbPath ?? "./data/memory-lancedb-brain");
      const embeddingConfig: EmbeddingConfig = {
        ...DEFAULT_EMBEDDING,
        ...(config.embedding ?? {}),
      };

      const storage = await MemoryStorage.connect(dbPath);
      const embedder = createEmbedder(embeddingConfig);
      const owners = normalizeOwners(config.owners);
      const agentWhitelist = config.agentWhitelist ?? DEFAULT_WHITELIST;

      registerAllMemoryTools(api, {
        storage,
        embedder,
        owners,
        agentWhitelist,
        retrieval: config.retrieval,
      });

      (api as any).__memoryLanceDbBrain = {
        storage,
        embedder,
        owners,
        agentWhitelist,
        config,
      };

      api.logger?.info?.(`memory-lancedb-brain: Phase 2 initialized (db=${dbPath})`);
    } catch (error) {
      api.logger?.error?.(`memory-lancedb-brain: initialization failed: ${String(error)}`);
      throw error;
    }
  })();
}
