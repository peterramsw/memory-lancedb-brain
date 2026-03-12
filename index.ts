/**
 * memory-lancedb-brain - OpenClaw intelligent memory plugin
 * Phase 3: tools + context engine + manual distill command
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { createDistiller, type DistillerConfig } from "./src/distill.js";
import {
  createMemoryBrainContextEngine,
  createMemoryDistillCommand,
  type SessionState,
} from "./src/context-engine.js";
import { createEmbedder, type EmbeddingConfig } from "./src/embedding.js";
import { normalizeOwners, resolveOwnerFromContext } from "./src/owners.js";
import { MemoryStorage } from "./src/storage.js";
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

function upsertSessionState(
  sessionStates: Map<string, SessionState>,
  sessionKeyIndex: Map<string, string>,
  lastSessionByChannel: Map<string, string>,
  partial: Partial<SessionState> & { sessionId: string },
): SessionState {
  const existing = sessionStates.get(partial.sessionId);
  const next: SessionState = {
    sessionId: partial.sessionId,
    sessionKey: partial.sessionKey ?? existing?.sessionKey,
    agentId: partial.agentId ?? existing?.agentId,
    owner: partial.owner ?? existing?.owner,
    channelId: partial.channelId ?? existing?.channelId,
    sessionFile: partial.sessionFile ?? existing?.sessionFile,
    staging: partial.staging ?? existing?.staging ?? [],
    childSessionKeys: partial.childSessionKeys ?? existing?.childSessionKeys ?? [],
    parentSessionKey: partial.parentSessionKey ?? existing?.parentSessionKey,
    subagentEnded: partial.subagentEnded ?? existing?.subagentEnded ?? false,
    updatedAt: Date.now(),
  };
  sessionStates.set(partial.sessionId, next);
  if (next.sessionKey) sessionKeyIndex.set(next.sessionKey, next.sessionId);
  if (next.channelId) lastSessionByChannel.set(next.channelId, next.sessionId);
  return next;
}

export default function register(api: OpenClawPluginApi & { logger?: any; pluginConfig?: unknown; on?: Function }): void {
  void (async () => {
    try {
      const config = ((api.pluginConfig ?? {}) as PluginConfig) ?? {};
      const dbPath = api.resolvePath?.(config.dbPath ?? "./data/memory-lancedb-brain") ?? (config.dbPath ?? "./data/memory-lancedb-brain");
      const embeddingConfig: EmbeddingConfig = {
        ...DEFAULT_EMBEDDING,
        ...(config.embedding ?? {}),
      };
      const distillationConfig: DistillerConfig = {
        model: config.distillation?.model ?? "vllm/Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
        baseURL: config.distillation?.baseURL ?? "http://127.0.0.1:32080/v1",
        apiKey: config.distillation?.apiKey ?? "local",
      };

      const storage = await MemoryStorage.connect(dbPath);
      const embedder = createEmbedder(embeddingConfig);
      const distiller = createDistiller(distillationConfig);
      const owners = normalizeOwners(config.owners);
      const agentWhitelist = config.agentWhitelist ?? DEFAULT_WHITELIST;
      const sessionStates = new Map<string, SessionState>();
      const sessionKeyIndex = new Map<string, string>();
      const lastSessionByChannel = new Map<string, string>();

      registerAllMemoryTools(api, {
        storage,
        embedder,
        owners,
        agentWhitelist,
        retrieval: config.retrieval,
      });

      const engine = createMemoryBrainContextEngine({
        storage,
        embedder,
        distiller,
        owners,
        agentWhitelist,
        retrieval: config.retrieval,
        sessionStates,
        sessionKeyIndex,
        lastSessionByChannel,
      });

      api.registerContextEngine?.("memory-lancedb-brain", () => engine);
      api.registerCommand?.(createMemoryDistillCommand({
        storage,
        embedder,
        distiller,
        owners,
        agentWhitelist,
        retrieval: config.retrieval,
        sessionStates,
        sessionKeyIndex,
        lastSessionByChannel,
      }, engine));

      api.on?.("before_prompt_build", (_event: unknown, ctx: any) => {
        if (!ctx?.sessionId) return;
        const owner = resolveOwnerFromContext(
          {
            senderId: undefined,
            messageChannel: ctx.channelId,
            agentId: ctx.agentId,
            senderIsOwner: true,
          },
          owners,
        ) ?? (owners[0] ? { ownerId: owners[0].owner_id, ownerNamespace: owners[0].owner_namespace } : undefined);

        upsertSessionState(sessionStates, sessionKeyIndex, lastSessionByChannel, {
          sessionId: ctx.sessionId,
          sessionKey: ctx.sessionKey,
          agentId: ctx.agentId,
          owner,
          channelId: ctx.channelId,
        });
      });

      api.on?.("session_start", (event: any, ctx: any) => {
        if (!event?.sessionId) return;
        upsertSessionState(sessionStates, sessionKeyIndex, lastSessionByChannel, {
          sessionId: event.sessionId,
          sessionKey: event.sessionKey ?? ctx?.sessionKey,
          agentId: ctx?.agentId,
          channelId: ctx?.channelId,
          owner: owners[0] ? { ownerId: owners[0].owner_id, ownerNamespace: owners[0].owner_namespace } : undefined,
        });
      });

      api.on?.("subagent_spawned", (event: any) => {
        if (!event?.childSessionKey || !event?.runId) return;
        const sessionId = sessionKeyIndex.get(event.childSessionKey) ?? event.childSessionKey;
        upsertSessionState(sessionStates, sessionKeyIndex, lastSessionByChannel, {
          sessionId,
          sessionKey: event.childSessionKey,
          agentId: event.agentId,
          owner: owners[0] ? { ownerId: owners[0].owner_id, ownerNamespace: owners[0].owner_namespace } : undefined,
        });
      });

      (api as any).__memoryLanceDbBrain = {
        storage,
        embedder,
        distiller,
        owners,
        agentWhitelist,
        config,
        engine,
        sessionStates,
        sessionKeyIndex,
        lastSessionByChannel,
      };

      api.logger?.info?.(`memory-lancedb-brain: Phase 3 initialized (db=${dbPath})`);
    } catch (error) {
      api.logger?.error?.(`memory-lancedb-brain: initialization failed: ${String(error)}`);
      throw error;
    }
  })();
}
