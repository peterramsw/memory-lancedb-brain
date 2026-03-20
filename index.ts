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
  type BrainDiagnostics,
  type SessionState,
} from "./src/context-engine.js";
import { createEmbedder, type EmbeddingConfig } from "./src/embedding.js";
import { normalizeOwners, resolveOwnerFromContext, type ResolvedOwner } from "./src/owners.js";
import { MemoryStorage } from "./src/storage.js";
import { registerAllMemoryTools } from "./src/tools.js";

// Process-global state so duplicate plugin inits share maps, storage, and engine.
const GLOBAL_STATE_KEY = Symbol.for("memory-lancedb-brain.sessionState");
const PLUGIN_ID = "memory-lancedb-brain";
const PLUGIN_VERSION = "0.2.3";
type GlobalState = {
  sessionStates: Map<string, SessionState>;
  sessionKeyIndex: Map<string, string>;
  lastSessionByChannel: Map<string, string>;
  eventsRegistered: boolean;
  // Mutable refs updated on each re-init so event handlers use latest instances
  engine: ReturnType<typeof createMemoryBrainContextEngine> | null;
  storage: MemoryStorage | null;
  diagnostics: BrainDiagnostics | null;
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
      diagnostics: null,
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
    messageChannel: partial.messageChannel ?? existing?.messageChannel,
    requesterSenderId: partial.requesterSenderId ?? existing?.requesterSenderId,
    agentAccountId: partial.agentAccountId ?? existing?.agentAccountId,
    senderIsOwner: partial.senderIsOwner ?? existing?.senderIsOwner,
    channelId: partial.channelId ?? existing?.channelId,
    pendingPrompt: partial.pendingPrompt ?? existing?.pendingPrompt,
    pendingPromptContext: partial.pendingPromptContext ?? existing?.pendingPromptContext,
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

function createDiagnostics(dbPath: string): BrainDiagnostics {
  return {
    pluginId: PLUGIN_ID,
    dbPath,
    initializedAt: Date.now(),
    recentWarnings: [],
    recentErrors: [],
  };
}

function pushRecent(list: Array<{ at: number; message: string }>, message: string, max = 8): void {
  list.push({ at: Date.now(), message });
  if (list.length > max) list.splice(0, list.length - max);
}

async function detectTrustedPluginAllowance(pluginId: string): Promise<boolean | undefined> {
  try {
    const home = process.env.HOME;
    if (!home) return undefined;
    const { readFile } = await import("node:fs/promises");
    const content = await readFile(join(home, ".openclaw", "openclaw.json"), "utf8");
    const parsed = JSON.parse(content);
    const allow = parsed?.plugins?.allow;
    if (!Array.isArray(allow)) return undefined;
    return allow.includes(pluginId);
  } catch {
    return undefined;
  }
}

function resolveHookContext(ctx: any = {}, event: any = {}) {
  return {
    sessionId: event?.sessionId ?? ctx?.sessionId,
    sessionKey: event?.sessionKey ?? ctx?.sessionKey,
    agentId: event?.agentId ?? ctx?.agentId,
    channelId: ctx?.channelId,
    messageChannel: ctx?.messageChannel,
    requesterSenderId: ctx?.requesterSenderId,
    agentAccountId: ctx?.agentAccountId,
    senderIsOwner: typeof ctx?.senderIsOwner === "boolean" ? ctx.senderIsOwner : undefined,
  };
}

function resolveHookOwner(
  owners: ReturnType<typeof normalizeOwners>,
  runtimeCtx: ReturnType<typeof resolveHookContext>,
): ResolvedOwner | undefined {
  return resolveOwnerFromContext(
    {
      senderId: runtimeCtx.requesterSenderId ?? runtimeCtx.agentAccountId,
      messageChannel: runtimeCtx.messageChannel,
      agentId: runtimeCtx.agentId,
      senderIsOwner: runtimeCtx.senderIsOwner,
    },
    owners,
  ) ?? (owners[0] ? { ownerId: owners[0].owner_id, ownerNamespace: owners[0].owner_namespace } : undefined);
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
      const diagnostics = gs.diagnostics ?? createDiagnostics(dbPath);
      diagnostics.dbPath = dbPath;
      diagnostics.trustedPluginExplicit = await detectTrustedPluginAllowance(PLUGIN_ID);
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
        diagnostics,
        dbPath,
        pluginId: PLUGIN_ID,
        pluginVersion: PLUGIN_VERSION,
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
        diagnostics,
      });

      // Update global refs so event handlers always use the latest instances
      gs.engine = engine;
      gs.storage = storage;
      gs.diagnostics = diagnostics;

      api.registerContextEngine?.(PLUGIN_ID, () => engine);
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
        diagnostics,
      }, engine));

      // Only register event handlers once (plugin may re-initialize but api.on accumulates)
      if (!gs.eventsRegistered) {
        gs.eventsRegistered = true;

        const capturePendingPrompt = (event: any, ctx: any) => {
          const runtimeCtx = resolveHookContext(ctx);
          if (!runtimeCtx.sessionId) return;
          const owner = resolveHookOwner(owners, runtimeCtx);
          const prompt = typeof event?.prompt === "string" ? event.prompt : undefined;
          upsertSessionState(sessionStates, sessionKeyIndex, lastSessionByChannel, {
            sessionId: runtimeCtx.sessionId,
            sessionKey: runtimeCtx.sessionKey,
            agentId: runtimeCtx.agentId,
            owner,
            messageChannel: runtimeCtx.messageChannel,
            requesterSenderId: runtimeCtx.requesterSenderId,
            agentAccountId: runtimeCtx.agentAccountId,
            senderIsOwner: runtimeCtx.senderIsOwner,
            channelId: runtimeCtx.channelId,
            pendingPrompt: prompt,
          });
        };

        api.on?.("before_model_resolve", capturePendingPrompt);
        api.on?.("before_agent_start", capturePendingPrompt);

        api.on?.("before_prompt_build", (_event: unknown, ctx: any) => {
          const runtimeCtx = resolveHookContext(ctx);
          if (!runtimeCtx.sessionId) return;
          const owner = resolveHookOwner(owners, runtimeCtx);

          const session = upsertSessionState(sessionStates, sessionKeyIndex, lastSessionByChannel, {
            sessionId: runtimeCtx.sessionId,
            sessionKey: runtimeCtx.sessionKey,
            agentId: runtimeCtx.agentId,
            owner,
            messageChannel: runtimeCtx.messageChannel,
            requesterSenderId: runtimeCtx.requesterSenderId,
            agentAccountId: runtimeCtx.agentAccountId,
            senderIsOwner: runtimeCtx.senderIsOwner,
            channelId: runtimeCtx.channelId,
          });

          const prependContext = session.pendingPromptContext?.trim();
          if (!prependContext) return;
          session.pendingPromptContext = undefined;
          sessionStates.set(session.sessionId, session);
          return { prependContext };
        });

        api.on?.("session_start", (event: any, ctx: any) => {
          const runtimeCtx = resolveHookContext(ctx, event);
          if (!runtimeCtx.sessionId) return;
          const owner = resolveHookOwner(owners, runtimeCtx);
          upsertSessionState(sessionStates, sessionKeyIndex, lastSessionByChannel, {
            sessionId: runtimeCtx.sessionId,
            sessionKey: runtimeCtx.sessionKey,
            agentId: runtimeCtx.agentId,
            owner,
            messageChannel: runtimeCtx.messageChannel,
            requesterSenderId: runtimeCtx.requesterSenderId,
            agentAccountId: runtimeCtx.agentAccountId,
            senderIsOwner: runtimeCtx.senderIsOwner,
            channelId: runtimeCtx.channelId,
          });
        });

        api.on?.("subagent_spawned", (event: any, ctx: any) => {
          if (!event?.childSessionKey || !event?.runId) return;
          const sessionId = sessionKeyIndex.get(event.childSessionKey) ?? event.childSessionKey;
          const runtimeCtx = resolveHookContext(ctx, event);
          const owner = resolveHookOwner(owners, runtimeCtx);
          upsertSessionState(sessionStates, sessionKeyIndex, lastSessionByChannel, {
            sessionId,
            sessionKey: event.childSessionKey,
            agentId: runtimeCtx.agentId,
            owner,
            messageChannel: runtimeCtx.messageChannel,
            requesterSenderId: runtimeCtx.requesterSenderId,
            agentAccountId: runtimeCtx.agentAccountId,
            senderIsOwner: runtimeCtx.senderIsOwner,
            channelId: runtimeCtx.channelId,
          });
        });

        // Auto-distill when session ends (triggered by /new, /reset, idle timeout)
        api.on?.("session_end", async (event: any, ctx: any) => {
          const runtimeCtx = resolveHookContext(ctx, event);
          if (!runtimeCtx.sessionId) return;
          let session = sessionStates.get(runtimeCtx.sessionId);
          if (!session) {
            const owner = resolveHookOwner(owners, runtimeCtx);
            session = upsertSessionState(sessionStates, sessionKeyIndex, lastSessionByChannel, {
              sessionId: runtimeCtx.sessionId,
              sessionKey: runtimeCtx.sessionKey,
              agentId: runtimeCtx.agentId,
              owner,
              messageChannel: runtimeCtx.messageChannel,
              requesterSenderId: runtimeCtx.requesterSenderId,
              agentAccountId: runtimeCtx.agentAccountId,
              senderIsOwner: runtimeCtx.senderIsOwner,
              channelId: runtimeCtx.channelId,
            });
            const warn = `memory-lancedb-brain: session_end reconstructed missing session state for ${runtimeCtx.sessionId}`;
            pushRecent(diagnostics.recentWarnings, warn);
            api.logger?.warn?.(warn);
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
            const warn = `memory-lancedb-brain: session_end skipped — no sessionFile for ${runtimeCtx.sessionId} (sessionKey=${session.sessionKey})`;
            pushRecent(diagnostics.recentWarnings, warn);
            api.logger?.warn?.(warn);
            return;
          }

          try {
            const currentEngine = gs.engine;
            if (!currentEngine) {
              const warn = "memory-lancedb-brain: session_end skipped — engine not initialized";
              pushRecent(diagnostics.recentWarnings, warn);
              api.logger?.warn?.(warn);
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
              const warn = `memory-lancedb-brain: auto-distill on session_end no-op for ${event.sessionId}: ${result.reason ?? "no memories"}`;
              pushRecent(diagnostics.recentWarnings, warn);
              api.logger?.warn?.(warn);
            }
          } catch (err) {
            const warn = `memory-lancedb-brain: auto-distill failed on session_end: ${String(err)}`;
            pushRecent(diagnostics.recentErrors, warn);
            api.logger?.warn?.(warn);
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
        diagnostics,
      };

      if (diagnostics.trustedPluginExplicit === false) {
        pushRecent(
          diagnostics.recentWarnings,
          "trusted plugin loading is not explicit; plugins.allow does not include memory-lancedb-brain",
        );
      }
      api.logger?.info?.(`memory-lancedb-brain: Phase 3 initialized (db=${dbPath})`);
    } catch (error) {
      api.logger?.error?.(`memory-lancedb-brain: initialization failed: ${String(error)}`);
      throw error;
    }
  })();
}
