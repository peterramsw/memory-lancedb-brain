/**
 * memory-lancedb-brain - OpenClaw intelligent memory plugin
 * Phase 3: tools + context engine + manual distill command
 */

import { existsSync } from "node:fs";
import { join } from "node:path";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { createDistiller, createLLMCaller, type DistillerConfig } from "./src/distill.js";
import {
  createMemoryBrainContextEngine,
  createMemoryDistillCommand,
  type SessionState,
} from "./src/context-engine.js";
import { createEmbedder, type EmbeddingConfig } from "./src/embedding.js";
import { normalizeOwners, resolveOwnerFromContext } from "./src/owners.js";
import { MemoryStorage } from "./src/storage.js";
import { registerAllMemoryTools } from "./src/tools.js";

// Process-global state so duplicate plugin inits share maps, storage, and engine.
const GLOBAL_STATE_KEY = Symbol.for("memory-lancedb-brain.sessionState");
type GlobalState = {
  sessionStates: Map<string, SessionState>;
  sessionKeyIndex: Map<string, string>;
  lastSessionByChannel: Map<string, string>;
  eventsRegistered: boolean;
  // Mutable refs updated on each re-init so event handlers use latest instances
  engine: ReturnType<typeof createMemoryBrainContextEngine> | null;
  storage: MemoryStorage | null;
};
function getGlobalState(): GlobalState {
  const g = globalThis as typeof globalThis & { [GLOBAL_STATE_KEY]?: GlobalState };
  if (!g[GLOBAL_STATE_KEY]) {
    g[GLOBAL_STATE_KEY] = {
      sessionStates: new Map(),
      sessionKeyIndex: new Map(),
      lastSessionByChannel: new Map(),
      eventsRegistered: false,
      engine: null,
      storage: null,
    };
  }
  return g[GLOBAL_STATE_KEY];
}

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
  autoDistill?: {
    enabled?: boolean;
    triggers?: Array<"onSubagentEnded" | "onSessionEnd" | "onReset" | "onNew">;
    minStagingLength?: number;
    tokenBudget?: number;
    onSubagentEnded?: boolean;
    onSessionEnd?: boolean;
    onReset?: boolean;
    onNew?: boolean;
  };
}

const DEFAULT_EMBEDDING: Required<EmbeddingConfig> = {
  apiKey: process.env.OPENCLAW_EMBEDDING_API_KEY ?? "local",
  model: process.env.OPENCLAW_EMBEDDING_MODEL ?? "text-embedding-3-small",
  baseURL: process.env.OPENCLAW_EMBEDDING_BASE_URL ?? "https://api.openai.com/v1",
  dimensions: Number(process.env.OPENCLAW_EMBEDDING_DIMENSIONS) || 1536,
};

const DEFAULT_RETRIEVAL = {
  mode: "hybrid" as const,
  rerank: false,  // off by default — user enables via config with their own endpoint
  rerankEndpoint: "",
  rerankApiKey: "",
  rerankModel: "",
};

const DEFAULT_WHITELIST = [
  "main",
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
        model: config.distillation?.model,   // falls through to createDistiller() defaults + env vars
        baseURL: config.distillation?.baseURL,
        apiKey: config.distillation?.apiKey,
      };

      const mergedRetrieval = { ...DEFAULT_RETRIEVAL, ...(config.retrieval ?? {}) };
      const gs = getGlobalState();
      // Reuse storage across plugin re-inits to avoid stale LanceDB table handles
      const storage = gs.storage ?? await MemoryStorage.connect(dbPath);
      const embedder = createEmbedder(embeddingConfig);
      const distiller = createDistiller(distillationConfig);
      const llmCaller = createLLMCaller(distillationConfig);
      const owners = normalizeOwners(config.owners);
      const agentWhitelist = config.agentWhitelist ?? DEFAULT_WHITELIST;
      const sessionStates = gs.sessionStates;
      const sessionKeyIndex = gs.sessionKeyIndex;
      const lastSessionByChannel = gs.lastSessionByChannel;

      registerAllMemoryTools(api, {
        storage,
        embedder,
        owners,
        agentWhitelist,
        retrieval: mergedRetrieval,
      });

      const engine = createMemoryBrainContextEngine({
        storage,
        embedder,
        distiller,
        llmCaller,
        owners,
        agentWhitelist,
        retrieval: mergedRetrieval,
        autoDistill: config.autoDistill as any,
        sessionStates,
        sessionKeyIndex,
        lastSessionByChannel,
      });

      // Update global refs so event handlers always use the latest instances
      gs.engine = engine;
      gs.storage = storage;

      api.registerContextEngine?.("memory-lancedb-brain", () => engine);
      api.registerCommand?.(createMemoryDistillCommand({
        storage,
        embedder,
        distiller,
        llmCaller,
        owners,
        agentWhitelist,
        retrieval: mergedRetrieval,
        sessionStates,
        sessionKeyIndex,
        lastSessionByChannel,
      }, engine));

      // Only register event handlers once (plugin may re-initialize but api.on accumulates)
      if (!gs.eventsRegistered) {
        gs.eventsRegistered = true;

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

        // Auto-distill when session ends (triggered by /new, /reset, idle timeout)
        api.on?.("session_end", async (event: any, _ctx: any) => {
          if (!event?.sessionId) return;
          const session = sessionStates.get(event.sessionId);
          if (!session) {
            api.logger?.warn?.(`memory-lancedb-brain: session_end ignored — no session state for ${event.sessionId}`);
            return;
          }

          // Resolve sessionFile: prefer state, fallback to filesystem
          let sessionFile = session.sessionFile;
          if (!sessionFile) {
            const agentId = session.agentId ?? "main";
            const candidate = join(
              process.env.HOME ?? "",
              ".openclaw/agents",
              agentId,
              "sessions",
              `${event.sessionId}.jsonl`,
            );
            if (existsSync(candidate)) {
              sessionFile = candidate;
              api.logger?.info?.(`memory-lancedb-brain: session_end resolved sessionFile from filesystem: ${candidate}`);
            }
          }

          if (!sessionFile) {
            api.logger?.warn?.(`memory-lancedb-brain: session_end skipped — no sessionFile for ${event.sessionId} (sessionKey=${session.sessionKey})`);
            return;
          }

          try {
            const currentEngine = gs.engine;
            if (!currentEngine) {
              api.logger?.warn?.(`memory-lancedb-brain: session_end skipped — engine not initialized`);
              return;
            }
            const result = await currentEngine.compact({
              sessionId: session.sessionId,
              sessionFile,
              force: true,
            });
            if (result.ok && result.compacted) {
              api.logger?.info?.(`memory-lancedb-brain: auto-distill on session_end OK for ${event.sessionId}: inserted ${result.result?.insertedCount ?? 0} memories`);
            } else {
              api.logger?.warn?.(`memory-lancedb-brain: auto-distill on session_end no-op for ${event.sessionId}: ${result.reason ?? "no memories"}`);
            }
          } catch (err) {
            api.logger?.warn?.(`memory-lancedb-brain: auto-distill failed on session_end: ${err}`);
          }
        });
      }

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
