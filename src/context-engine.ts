/**
 * Context engine implementation for memory-lancedb-brain
 * Phase 5: Auto-distill with session lifecycle hooks
 */

import { readFile, readdir } from "node:fs/promises";
import { dirname, basename, join } from "node:path";
import type { MemoryRecord, MemoryEventRecord, MemoryScope, MemoryType } from "./schema.js";
import type { MemoryStorage } from "./storage.js";
import type { OwnerConfig, ResolvedOwner } from "./owners.js";
import type { DistillOutput } from "./distill.js";
import { buildCandidateMemories, processLifecycle } from "./lifecycle.js";
import type { Embedder } from "./embedding.js";

export interface AgentMessageLike {
  role?: string;
  content?: unknown;
  text?: string;
  type?: string;
  source?: string;
  name?: string;
}

export interface SessionState {
  sessionId: string;
  sessionKey?: string;
  sessionFile?: string;
  agentId?: string;
  owner?: ResolvedOwner;
  channelId?: string;
  staging: string[];
  childSessionKeys: string[];
  parentSessionKey?: string;
  subagentEnded?: boolean;
  updatedAt: number;
}

export interface AutoDistillConfig {
  enabled: boolean;
  triggers: Array<"onSubagentEnded" | "onSessionEnd" | "onReset" | "onNew">;
  minStagingLength: number;
  tokenBudget: number;
  onSubagentEnded: boolean;
  onSessionEnd: boolean;
  onReset: boolean;
  onNew: boolean;
}

export interface ContextEngineDeps {
  storage: MemoryStorage;
  embedder: Embedder;
  distiller: { distillTranscript(transcript: string, opts?: { customInstructions?: string }): Promise<DistillOutput> };
  owners: OwnerConfig[];
  agentWhitelist: string[];
  retrieval?: {
    mode?: "hybrid" | "vector" | "keyword";
    vectorWeight?: number;
    bm25Weight?: number;
    minScore?: number;
    hardMinScore?: number;
    rerank?: boolean;
    rerankApiKey?: string;
    rerankEndpoint?: string;
    rerankModel?: string;
    candidatePoolSize?: number;
  };
  autoDistill?: AutoDistillConfig;
  sessionStates: Map<string, SessionState>;
  sessionKeyIndex: Map<string, string>;
  lastSessionByChannel: Map<string, string>;
}

export interface AssembleResult {
  messages: AgentMessageLike[];
  estimatedTokens: number;
  systemPromptAddition?: string;
}

export interface CompactResult {
  ok: boolean;
  compacted: boolean;
  result?: {
    summary?: string;
    insertedCount?: number;
    details?: Record<string, unknown>;
  };
  reason?: string;
}

// Default config
const DEFAULT_AUTO_DISTILL: AutoDistillConfig = {
  enabled: true,
  triggers: ["onSubagentEnded", "onSessionEnd", "onReset", "onNew"],
  minStagingLength: 3,
  tokenBudget: 30000,
  onSubagentEnded: true,
  onSessionEnd: true,
  onReset: true,
  onNew: true,
};

// Helper functions
function upsertSessionState(
  deps: ContextEngineDeps,
  params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile?: string;
    agentId?: string;
    ownerId?: string;
    ownerNamespace?: string;
    channelId?: string;
    parentSessionKey?: string;
  },
): SessionState {
  const existing = deps.sessionStates.get(params.sessionId);
  if (existing) {
    // Update mutable fields on existing state
    if (params.sessionFile) existing.sessionFile = params.sessionFile;
    if (params.sessionKey) existing.sessionKey = params.sessionKey;
    if (params.agentId) existing.agentId = params.agentId;
    if (params.channelId) existing.channelId = params.channelId;
    if (params.ownerId && params.ownerNamespace) {
      existing.owner = { ownerId: params.ownerId, ownerNamespace: params.ownerNamespace };
    }
    existing.updatedAt = Date.now();
    return existing;
  }

  const newState: SessionState = {
    sessionId: params.sessionId,
    sessionKey: params.sessionKey,
    sessionFile: params.sessionFile,
    agentId: params.agentId,
    owner: params.ownerId && params.ownerNamespace
      ? { ownerId: params.ownerId, ownerNamespace: params.ownerNamespace }
      : undefined,
    channelId: params.channelId,
    staging: [],
    childSessionKeys: [],
    parentSessionKey: params.parentSessionKey,
    subagentEnded: false,
    updatedAt: Date.now(),
  };

  deps.sessionStates.set(params.sessionId, newState);

  if (params.sessionKey) {
    deps.sessionKeyIndex.set(params.sessionKey, params.sessionId);
  }

  return newState;
}

function extractMessageText(message: AgentMessageLike | any): string {
  if (!message) return "";
  if (typeof message === "string") return message;
  if (message.text) return typeof message.text === "string" ? message.text : String(message.text);
  if (message.content) {
    const content = message.content;
    if (Array.isArray(content)) {
      const textBlocks = content
        .filter((c: any) => c?.type === "text" && typeof c?.text === "string")
        .map((c: any) => c.text);
      if (textBlocks.length > 0) return textBlocks.join(" ");
    }
    if (typeof content === "string") return content;
  }
  return JSON.stringify(message);
}

function shouldStageText(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed || trimmed.length < 10) return false;
  if (trimmed.startsWith("/") && !trimmed.includes(":")) return false;
  if (trimmed.includes("<relevant-memories>") || trimmed.includes("[MEMORY_RECALL]")) return false;
  return true;
}

// Adaptive retrieval: skip embedding calls for greetings, commands, affirmations
const SKIP_RETRIEVAL_PATTERNS = [
  /^(hi|hello|hey|good\s*(morning|afternoon|evening)|你好|嗨|早安|晚安)\b/i,
  /^\//,
  /^(yes|no|ok|okay|sure|好|是|對|不|可以|了解|收到|👍|👎|✅|❌)\s*[.!]?$/i,
  /^(go ahead|continue|proceed|do it|繼續|開始|好的)\s*[.!]?$/i,
  /^[\p{Emoji}\s]+$/u,
  /^HEARTBEAT/i,
];

function shouldSkipRetrieval(query: string): boolean {
  const trimmed = query.trim();
  if (trimmed.length < 4) return true;
  // Force retrieve for memory-intent queries
  if (/(remember|recall|之前|上次|以前|還記得|記得|提到過|說過|last time|previously)/i.test(trimmed)) return false;
  if (SKIP_RETRIEVAL_PATTERNS.some(p => p.test(trimmed))) return true;
  // Skip very short non-question messages
  const hasCJK = /[\u4e00-\u9fff]/.test(trimmed);
  if (trimmed.length < (hasCJK ? 6 : 15) && !trimmed.includes('?') && !trimmed.includes('？')) return true;
  return false;
}

// Post-retrieval scoring: recency boost + importance weighting
function applyPostScoring(memories: MemoryRecord[]): MemoryRecord[] {
  if (memories.length <= 1) return memories;
  const now = Date.now();
  const scored = memories.map(m => {
    // Recency boost: half-life 14 days, max +0.15 bonus
    const ageDays = (now - (m.updated_at || m.created_at)) / 86_400_000;
    const recencyBoost = Math.exp(-ageDays / 14) * 0.15;
    // Importance weight: brain uses 1-5 scale, normalize to 0-1
    const importanceWeight = 0.7 + 0.3 * ((m.importance - 1) / 4);
    const score = (1 + recencyBoost) * importanceWeight;
    return { memory: m, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.map(s => s.memory);
}

async function selectRelevantMemories(
  deps: ContextEngineDeps,
  session: SessionState,
  scope: "owner_shared" | "agent_local",
  limit: number,
  query?: string,
): Promise<MemoryRecord[]> {
  try {
    const filters: Record<string, unknown> = { memory_scope: scope as MemoryScope, status: "active" as const };
    // agent_local memories should only be visible to the agent that created them
    if (scope === "agent_local" && session.agentId) {
      filters.agent_id = session.agentId;
    }
    let results: MemoryRecord[];

    if (query && deps.embedder) {
      // Vector search — fetch 3x candidates then re-rank
      const embedding = await deps.embedder.embed(query);
      results = await deps.storage.vectorSearch(embedding, limit * 3, filters);
    } else {
      results = await deps.storage.queryMemoriesByFilter(filters);
    }

    // Apply recency + importance scoring, then trim
    return applyPostScoring(results).slice(0, limit);
  } catch (error) {
    console.error(`[memory-lancedb-brain] selectRelevantMemories error (scope=${scope}): ${error instanceof Error ? error.stack : String(error)}`);
    return [];
  }
}

function trimMemoriesByTokenBudget(memories: MemoryRecord[], budget: number): MemoryRecord[] {
  let totalTokens = 0;
  const result: MemoryRecord[] = [];

  for (const mem of memories) {
    const tokens = Math.ceil(mem.content.length / 4);
    if (totalTokens + tokens > budget) break;
    totalTokens += tokens;
    result.push(mem);
  }

  return result;
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function renderMemorySection(title: string, memories: MemoryRecord[]): string {
  if (memories.length === 0) return "";
  const items = memories.map(m => `- ${m.content}`).join("\n");
  return `### ${title}\n\n${items}`;
}

async function compactSession(
  deps: ContextEngineDeps,
  params: {
    sessionId: string;
    sessionFile: string;
    currentTokenCount?: number;
    force?: boolean;
    customInstructions?: string;
  },
): Promise<CompactResult> {
  try {
    const session = deps.sessionStates.get(params.sessionId);

    // Resolve session file: try original path first, then .reset.* rename
    let resolvedFile = params.sessionFile;
    let sessionContent: string;
    try {
      sessionContent = await readFile(resolvedFile, "utf-8");
    } catch {
      // openclaw renames session files to .jsonl.reset.<timestamp> on /new
      const dir = dirname(params.sessionFile);
      const base = basename(params.sessionFile);
      const files = await readdir(dir);
      const resetFile = files
        .filter(f => f.startsWith(base + ".reset."))
        .sort()
        .at(-1); // most recent
      if (resetFile) {
        resolvedFile = join(dir, resetFile);
        console.log(`[memory-lancedb-brain] compactSession: resolved renamed file ${resolvedFile}`);
        sessionContent = await readFile(resolvedFile, "utf-8");
      } else {
        throw new Error(`Session file not found: ${params.sessionFile} (no .reset.* variant either)`);
      }
    }
    const lines = sessionContent.split("\n").filter(l => l.trim());

    if (lines.length === 0) {
      return { ok: true, compacted: false, reason: "No session transcript to distill" };
    }

    // Build conversation transcript from last 100 messages
    const transcriptParts: string[] = [];
    for (const line of lines.slice(-100)) {
      try {
        const entry = JSON.parse(line);
        if (entry?.type !== "message") continue;

        const role = entry.message?.role;
        const content = entry.message?.content;

        if (role !== "user" && role !== "assistant") continue;

        const text = typeof content === "string"
          ? content
          : Array.isArray(content)
            ? content.filter((c: any) => c?.type === "text").map((c: any) => c.text).join(" ")
            : JSON.stringify(content);

        if (text && text.trim().length > 10) {
          transcriptParts.push(`${role}: ${text}`);
        }
      } catch {
        // Ignore parse errors
      }
    }

    if (transcriptParts.length < 2) {
      return { ok: true, compacted: false, reason: "Insufficient conversation content" };
    }

    const fullTranscript = transcriptParts.join("\n");

    // Distill using LLM
    const distillResult = await deps.distiller.distillTranscript(fullTranscript, {
      customInstructions: params.customInstructions,
    });

    // Build candidate memories with proper embeddings via lifecycle.ts
    const ownerId = session?.owner?.ownerId ?? "peter";
    const ownerNamespace = session?.owner?.ownerNamespace ?? "peter";
    const agentId = session?.agentId ?? "unknown";

    const candidates = await buildCandidateMemories(
      { storage: deps.storage, embedder: deps.embedder },
      { sessionId: params.sessionId, ownerId, ownerNamespace, agentId },
      distillResult,
    );

    if (candidates.length === 0) {
      return { ok: true, compacted: false, reason: "Distillation produced no memories" };
    }

    // Process through lifecycle pipeline (merge dedup + supersede detection + insert)
    const lifecycleResult = await processLifecycle(
      { storage: deps.storage, embedder: deps.embedder },
      candidates,
    );

    return {
      ok: true,
      compacted: true,
      result: {
        summary: distillResult.session_summary || "Session distilled",
        insertedCount: lifecycleResult.inserted,
        details: {
          insertedCount: lifecycleResult.inserted,
          merged: lifecycleResult.merged,
          supersedeCandidates: lifecycleResult.supersedeCandidates,
          insertedTypes: lifecycleResult.insertedTypes,
        },
      },
    };
  } catch (error) {
    return {
      ok: false,
      compacted: false,
      reason: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

async function checkAndAutoDistill(
  deps: ContextEngineDeps,
  session: SessionState,
  trigger: string,
): Promise<boolean> {
  const config = deps.autoDistill ?? DEFAULT_AUTO_DISTILL;

  if (!config.enabled) {
    return false;
  }

  if (!session.sessionFile) {
    console.log(`[memory-lancedb-brain] Auto-distill skipped (${trigger}): no sessionFile for session ${session.sessionId}`);
    return false;
  }

  if (session.staging.length < config.minStagingLength) {
    console.log(`[memory-lancedb-brain] Auto-distill skipped (${trigger}): staging ${session.staging.length} < min ${config.minStagingLength}`);
    return false;
  }

  console.log(`[memory-lancedb-brain] Auto-distill triggered by: ${trigger}, staging length: ${session.staging.length}`);

  try {
    const result = await compactSession(deps, {
      sessionId: session.sessionId,
      sessionFile: session.sessionFile,
      force: true,
    });

    if (result.ok && result.compacted) {
      console.log(`[memory-lancedb-brain] Auto-distill success (${trigger}): inserted ${result.result?.insertedCount ?? 0} memories`);
      session.staging = [];
      deps.sessionStates.set(session.sessionId, session);
      return true;
    }

    console.log(`[memory-lancedb-brain] Auto-distill no-op (${trigger}): ${result.reason ?? "no memories produced"}`);
    return false;
  } catch (error) {
    console.error(`[memory-lancedb-brain] Auto-distill failed (${trigger}): ${error instanceof Error ? error.message : String(error)}`);
    return false;
  }
}

export function createMemoryBrainContextEngine(deps: ContextEngineDeps) {
  return {
    info: {
      id: "memory-lancedb-brain",
      name: "Memory LanceDB Brain",
      ownsCompaction: false,
    },

    async bootstrap(params: { sessionId: string; sessionFile: string }) {
      console.log(`[memory-lancedb-brain] bootstrap: sessionId=${params.sessionId} sessionFile=${params.sessionFile}`);
      upsertSessionState(deps, {
        sessionId: params.sessionId,
        sessionKey: params.sessionFile.replace(/\.jsonl$/, ""),
        sessionFile: params.sessionFile,
      });
      return { bootstrapped: true, importedMessages: 0 };
    },

    async ingest(params: { sessionId: string; message: AgentMessageLike; isHeartbeat?: boolean }) {
      if (params.isHeartbeat) return { ingested: false };
      const session = upsertSessionState(deps, { sessionId: params.sessionId });
      const text = extractMessageText(params.message);
      if (!shouldStageText(text)) return { ingested: false };
      session.staging.push(text);
      session.staging = session.staging.slice(-20);
      session.updatedAt = Date.now();
      deps.sessionStates.set(session.sessionId, session);
      return { ingested: true };
    },

    async afterTurn(params: {
      sessionId: string;
      sessionFile: string;
      messages: AgentMessageLike[];
      prePromptMessageCount: number;
      autoCompactionSummary?: string;
    }) {
      console.log(`[memory-lancedb-brain] afterTurn: sessionId=${params.sessionId} sessionFile=${params.sessionFile} messages=${params.messages.length}`);
      const session = upsertSessionState(deps, {
        sessionId: params.sessionId,
        sessionKey: params.sessionFile.replace(/\.jsonl$/, ""),
        sessionFile: params.sessionFile,
      });
      for (const message of params.messages.slice(params.prePromptMessageCount)) {
        const text = extractMessageText(message);
        if (shouldStageText(text)) session.staging.push(text);
      }
      if (params.autoCompactionSummary && shouldStageText(params.autoCompactionSummary)) {
        session.staging.push(params.autoCompactionSummary);
      }
      session.staging = session.staging.slice(-30);
      deps.sessionStates.set(session.sessionId, session);
    },

    async assemble(params: { sessionId: string; messages: AgentMessageLike[]; tokenBudget?: number }): Promise<AssembleResult> {
      const session = upsertSessionState(deps, { sessionId: params.sessionId });
      const recentStaging = session.staging.slice(-5).join(" ").slice(0, 300);

      // Adaptive: skip retrieval for greetings, commands, short affirmations
      if (!recentStaging || shouldSkipRetrieval(recentStaging)) {
        return { messages: params.messages, estimatedTokens: 0 };
      }

      const ownerShared = trimMemoriesByTokenBudget(await selectRelevantMemories(deps, session, "owner_shared", 15, recentStaging), 2000);
      const agentLocal = trimMemoriesByTokenBudget(await selectRelevantMemories(deps, session, "agent_local", 5, recentStaging), 800);
      console.log(`[memory-lancedb-brain] assemble: sessionId=${params.sessionId} query=${recentStaging.slice(0, 60)}... ownerShared=${ownerShared.length} agentLocal=${agentLocal.length}`);

      if (ownerShared.length === 0 && agentLocal.length === 0) {
        return { messages: params.messages, estimatedTokens: 0 };
      }

      const sections = [
        renderMemorySection("OWNER SHARED MEMORY", ownerShared),
        renderMemorySection("AGENT LOCAL MEMORY", agentLocal),
      ].filter(Boolean);

      const memoryBlock = [
        "<long-term-memory>",
        "Facts about this user from previous conversations. Use them proactively to provide personalized context.",
        "Do NOT announce that you remember things or say \"I've noted that\". Just use the knowledge naturally.",
        "",
        sections.join("\n\n"),
        "</long-term-memory>",
      ].join("\n");

      return {
        messages: params.messages,
        estimatedTokens: estimateTokens(memoryBlock),
        systemPromptAddition: memoryBlock,
      };
    },

    async compact(params: {
      sessionId: string;
      sessionFile: string;
      tokenBudget?: number;
      force?: boolean;
      currentTokenCount?: number;
      customInstructions?: string;
    }): Promise<CompactResult> {
      return compactSession(deps, {
        sessionId: params.sessionId,
        sessionFile: params.sessionFile,
        currentTokenCount: params.currentTokenCount,
        force: params.force,
        customInstructions: params.customInstructions,
      });
    },

    async prepareSubagentSpawn(params: { parentSessionKey: string; childSessionKey: string; ttlMs?: number }) {
      const parentSessionId = deps.sessionKeyIndex.get(params.parentSessionKey);
      if (!parentSessionId) return undefined;
      const parent = deps.sessionStates.get(parentSessionId);
      if (!parent) return undefined;

      parent.childSessionKeys.push(params.childSessionKey);
      deps.sessionStates.set(parent.sessionId, parent);

      const childSessionId = deps.sessionKeyIndex.get(params.childSessionKey) ?? params.childSessionKey;
      upsertSessionState(deps, {
        sessionId: childSessionId,
        sessionKey: params.childSessionKey,
        agentId: parent.agentId,
        ownerId: parent.owner?.ownerId,
        ownerNamespace: parent.owner?.ownerNamespace,
        channelId: parent.channelId,
        parentSessionKey: params.parentSessionKey,
      });

      return {
        rollback: async () => {
          const childId = deps.sessionKeyIndex.get(params.childSessionKey) ?? params.childSessionKey;
          deps.sessionStates.delete(childId);
          deps.sessionKeyIndex.delete(params.childSessionKey);
        },
      };
    },

    async onSubagentEnded(params: { childSessionKey: string; reason: "deleted" | "completed" | "swept" | "released" }) {
      const childSessionId = deps.sessionKeyIndex.get(params.childSessionKey) ?? params.childSessionKey;
      const child = deps.sessionStates.get(childSessionId);
      if (!child) return;
      child.subagentEnded = true;
      deps.sessionStates.set(child.sessionId, child);

      // Auto-distill on subagent ended if enabled
      const config = deps.autoDistill ?? DEFAULT_AUTO_DISTILL;
      if (config.onSubagentEnded) {
        await checkAndAutoDistill(deps, child, "onSubagentEnded");
      }

      if (!child.parentSessionKey) return;
      const parentSessionId = deps.sessionKeyIndex.get(child.parentSessionKey);
      if (!parentSessionId) return;
      const parent = deps.sessionStates.get(parentSessionId);
      if (!parent) return;

      if (child.staging.length > 0) {
        parent.staging.push(`[subagent:${params.reason}] ${child.staging.join(" | ")}`);
        parent.staging = parent.staging.slice(-30);
        deps.sessionStates.set(parent.sessionId, parent);
      }
    },
  };
}

export function createMemoryDistillCommand(
  deps: ContextEngineDeps,
  engine: ReturnType<typeof createMemoryBrainContextEngine>,
) {
  return {
    name: "memory",
    description: "Manual memory operations for memory-lancedb-brain: distill, recall, list, store.",
    acceptsArgs: true,
    handler: async (ctx: { args?: string; channel: string }) => {
      const args = (ctx.args ?? "").trim();
      if (!args) {
        return { 
          text: `Usage:\n  /memory distill — Force distill current session to LanceDB\n  /memory recall [query] — Search memories by query\n  /memory list [scope] — List memories by scope (owner_shared/agent_local/all)\n  /memory store [text] — Store a single memory immediately`,
        };
      }

      // Parse command and arguments
      const parts = args.split(/\s+/);
      const command = parts[0]?.toLowerCase();
      const subArgs = parts.slice(1).join(" ");

      switch (command) {
        case "distill": {
          const sessionId = deps.lastSessionByChannel.get(ctx.channel) ?? [...deps.sessionStates.keys()].at(-1);
          if (!sessionId) {
            return { text: "No active session available for distillation yet." };
          }

          const session = deps.sessionStates.get(sessionId);
          if (!session) {
            return { text: "Session not found in memory states." };
          }

          // We can't distill without sessionFile, so return appropriate message
          if (!session.sessionFile) {
            return { text: "Cannot distill: no session transcript file available. Please complete some conversation first." };
          }

          const result = await engine.compact({
            sessionId,
            sessionFile: session.sessionFile,
            force: true,
            currentTokenCount: 0,
          });

          if (!result.ok) {
            return { text: `Distillation failed: ${result.reason ?? "unknown error"}`, isError: true };
          }

          const details = (result.result?.details ?? {}) as Record<string, unknown>;
          return {
            text: `Distilled session ${sessionId}: inserted ${details.insertedCount ?? 0} memories.`,
          };
        }

        case "recall": {
          if (!subArgs || subArgs.trim().length < 3) {
            return { text: "Usage: /memory recall [query]\nEnter a search query to find relevant memories." };
          }

          const sessionId = deps.lastSessionByChannel.get(ctx.channel) ?? [...deps.sessionStates.keys()].at(-1);
          if (!sessionId) {
            return { text: "No active session available for recall." };
          }

          const session = deps.sessionStates.get(sessionId);
          if (!session) {
            return { text: "Session not found in memory states." };
          }

          try {
            // Select memories from both scopes
            const ownerShared = await selectRelevantMemories(deps, session, "owner_shared", 5, subArgs);
            const agentLocal = await selectRelevantMemories(deps, session, "agent_local", 3, subArgs);
            
            let response = "";
            if (ownerShared.length > 0) {
              response += "### OWNER SHARED MEMORY\n\n";
              for (const mem of ownerShared) {
                response += `- [${mem.memory_scope}] ${mem.content}\n`;
              }
            }
            if (agentLocal.length > 0) {
              response += "\n### AGENT LOCAL MEMORY\n\n";
              for (const mem of agentLocal) {
                response += `- [${mem.memory_scope}] ${mem.content}\n`;
              }
            }

            if (!response) {
              return { text: "No relevant memories found for your query." };
            }

            return { text: response.trim() };
          } catch (error) {
            return { text: `Recall failed: ${error instanceof Error ? error.message : String(error)}`, isError: true };
          }
        }

        case "list": {
          const scopeFilter = subArgs?.toLowerCase() || "all";
          const validScopes = ["owner_shared", "agent_local", "all"];
          if (!validScopes.includes(scopeFilter)) {
            return { text: `Invalid scope: ${scopeFilter}. Use: owner_shared, agent_local, or all.` };
          }

          const sessionId = deps.lastSessionByChannel.get(ctx.channel) ?? [...deps.sessionStates.keys()].at(-1);
          if (!sessionId) {
            return { text: "No active session available for listing." };
          }

          const session = deps.sessionStates.get(sessionId);
          if (!session) {
            return { text: "Session not found in memory states." };
          }

          try {
            let results: MemoryRecord[] = [];
            
            if (scopeFilter === "all" || scopeFilter === "owner_shared") {
              const shared = await selectRelevantMemories(deps, session, "owner_shared", 20, undefined);
              results = results.concat(shared);
            }
            if (scopeFilter === "all" || scopeFilter === "agent_local") {
              const local = await selectRelevantMemories(deps, session, "agent_local", 20, undefined);
              results = results.concat(local);
            }

            if (results.length === 0) {
              return { text: `No memories found${scopeFilter !== "all" ? ` in scope: ${scopeFilter}` : ""}. Try /memory distill first to save conversation context.` };
            }

            let response = `### MEMORIES (${results.length} total)\n\n`;
            for (const mem of results.slice(0, 20)) {
              response += `- [${mem.memory_scope}] ${mem.content}\n`;
            }

            return { text: response.trim() };
          } catch (error) {
            return { text: `List failed: ${error instanceof Error ? error.message : String(error)}`, isError: true };
          }
        }

        case "store": {
          if (!subArgs || subArgs.trim().length < 3) {
            return { text: "Usage: /memory store [text]\nEnter a short statement to memorize immediately." };
          }

          const sessionId = deps.lastSessionByChannel.get(ctx.channel) ?? [...deps.sessionStates.keys()].at(-1);
          if (!sessionId) {
            return { text: "No active session available." };
          }

          const session = deps.sessionStates.get(sessionId);
          if (!session) {
            return { text: "Session not found." };
          }

          try {
            // Determine appropriate scope based on who is speaking
            const targetScope: MemoryScope = session.owner?.ownerNamespace === "agent_local" ? "agent_local" : "owner_shared";
            const now = Date.now();

            // Get agent ID from session or fallback
            const agentId = session.agentId || "unknown";
            const ownerId = session.owner?.ownerId || "peter";
            const ownerNamespace = session.owner?.ownerNamespace || "peter";

            // Generate embedding so vector search can find this memory
            const embedding = await deps.embedder.embed(subArgs.trim());

            const memory: MemoryRecord = {
              memory_id: crypto.randomUUID(),
              owner_namespace: ownerNamespace,
              owner_id: ownerId,
              agent_id: agentId,
              memory_scope: targetScope,
              memory_type: "preference" as MemoryType,
              title: subArgs.trim().slice(0, 50),
              content: subArgs.trim(),
              summary: subArgs.trim(),
              tags: JSON.stringify(["manual"]),
              importance: 4,
              confidence: 1.0,
              status: "active",
              supersedes_id: "",
              created_at: now,
              updated_at: now,
              last_used_at: now,
              source_session_id: sessionId,
              embedding,
            };

            await deps.storage.insertMemory(memory);
            
            // Create event
            const event: MemoryEventRecord = {
              event_id: crypto.randomUUID(),
              memory_id: memory.memory_id,
              event_type: "create",
              event_time: now,
              details_json: JSON.stringify({
                source: "direct_user_input",
                manual_store: true,
              }),
            };
            await deps.storage.insertEvent(event);
            
            return { text: `Stored: "${subArgs.trim()}"\nSaved to: ${targetScope}\nID: ${memory.memory_id.slice(0, 8)}...` };
          } catch (error) {
            return { text: `Store failed: ${error instanceof Error ? error.message : String(error)}`, isError: true };
          }
        }

        default:
          return { 
            text: `Unknown command: ${command}.\n\nUsage:\n  /memory distill\n  /memory recall [query]\n  /memory list [scope]\n  /memory store [text]`,
          };
      }
    },
  };
}
