/**
 * Context engine implementation for memory-lancedb-brain
 */

import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import type { MemoryRecord, MemoryScope, MemoryType } from "./schema.js";
import type { MemoryStorage } from "./storage.js";
import type { OwnerConfig, ResolvedOwner } from "./owners.js";
import { hybridRetrieve, generateSummary, generateTitle } from "./retrieval.js";
import type { DistillOutput } from "./distill.js";

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
  agentId?: string;
  owner?: ResolvedOwner;
  channelId?: string;
  sessionFile?: string;
  staging: string[];
  childSessionKeys: string[];
  parentSessionKey?: string;
  subagentEnded?: boolean;
  updatedAt: number;
}

export interface ContextEngineDeps {
  storage: MemoryStorage;
  embedder: { embed(text: string): Promise<number[]> };
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
  reason?: string;
  result?: {
    summary?: string;
    firstKeptEntryId?: string;
    tokensBefore: number;
    tokensAfter?: number;
    details?: unknown;
  };
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function normalizeText(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") {
          const text = (item as Record<string, unknown>).text;
          return typeof text === "string" ? text : "";
        }
        return "";
      })
      .filter(Boolean)
      .join("\n")
      .trim();
  }
  if (value && typeof value === "object") {
    const text = (value as Record<string, unknown>).text;
    if (typeof text === "string") return text.trim();
  }
  return "";
}

function extractMessageText(message: AgentMessageLike): string {
  return normalizeText(message?.content) || normalizeText(message?.text) || "";
}

function shouldStageText(text: string): boolean {
  const normalized = text.trim();
  if (!normalized || normalized.length < 40) return false;
  if (/^(ok|yes|no|收到|好|thanks|謝謝)$/i.test(normalized)) return false;
  return true;
}

function upsertSessionState(deps: ContextEngineDeps, partial: Partial<SessionState> & { sessionId: string }): SessionState {
  const existing = deps.sessionStates.get(partial.sessionId);
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
  deps.sessionStates.set(partial.sessionId, next);
  if (next.sessionKey) deps.sessionKeyIndex.set(next.sessionKey, next.sessionId);
  if (next.channelId) deps.lastSessionByChannel.set(next.channelId, next.sessionId);
  return next;
}

function resolveOwnerForSession(deps: ContextEngineDeps, session: SessionState): ResolvedOwner | undefined {
  if (session.owner) return session.owner;
  const owner = deps.owners[0];
  return owner
    ? {
        ownerId: owner.owner_id,
        ownerNamespace: owner.owner_namespace,
      }
    : undefined;
}

async function selectRelevantMemories(
  deps: ContextEngineDeps,
  session: SessionState,
  scope: MemoryScope,
  limit: number,
): Promise<MemoryRecord[]> {
  const owner = resolveOwnerForSession(deps, session);
  if (!owner) return [];

  const querySeed = session.staging.slice(-4).join("\n").trim() || `${session.agentId ?? "agent"} memory context`;
  const results = await hybridRetrieve(deps.storage, deps.embedder, {
    query: querySeed,
    ownerId: owner.ownerId,
    ownerNamespace: owner.ownerNamespace,
    agentId: session.agentId,
    scope,
    limit,
    mode: deps.retrieval?.mode,
    vectorWeight: deps.retrieval?.vectorWeight,
    bm25Weight: deps.retrieval?.bm25Weight,
    minScore: deps.retrieval?.minScore ?? 0.15,
    hardMinScore: deps.retrieval?.hardMinScore ?? 0.05,
    rerank: deps.retrieval?.rerank,
    rerankApiKey: deps.retrieval?.rerankApiKey,
    rerankEndpoint: deps.retrieval?.rerankEndpoint,
    rerankModel: deps.retrieval?.rerankModel,
    candidatePoolSize: deps.retrieval?.candidatePoolSize,
  });

  for (const result of results) {
    await deps.storage.updateMemory(result.memory.memory_id, { last_used_at: Date.now() });
  }

  return results.map((result) => result.memory).slice(0, limit);
}

function trimMemoriesByTokenBudget(memories: MemoryRecord[], tokenBudget: number): MemoryRecord[] {
  const selected: MemoryRecord[] = [];
  let used = 0;
  for (const memory of memories) {
    const text = `[${memory.memory_type}] ${memory.title}\n${memory.summary || memory.content}`;
    const tokens = estimateTokens(text);
    if (used + tokens > tokenBudget) break;
    selected.push(memory);
    used += tokens;
  }
  return selected;
}

function renderMemorySection(title: string, memories: MemoryRecord[]): string {
  if (memories.length === 0) return "";
  return [
    `## ${title}`,
    ...memories.map((memory, index) => `${index + 1}. [${memory.memory_type}] ${memory.title}\n${memory.summary || memory.content}`),
  ].join("\n");
}

async function readTranscriptText(sessionFile: string, fallbackMessages: AgentMessageLike[], staging: string[]): Promise<string> {
  try {
    const raw = await readFile(sessionFile, "utf8");
    const lines = raw
      .split(/\n+/)
      .map((line) => line.trim())
      .filter(Boolean);

    const transcriptLines: string[] = [];
    for (const line of lines) {
      try {
        const parsed = JSON.parse(line);
        const role = String(parsed.role ?? parsed.type ?? parsed.source ?? "message");
        const text = normalizeText(parsed.content) || normalizeText(parsed.text) || normalizeText(parsed.message);
        if (text) transcriptLines.push(`${role}: ${text}`);
      } catch {
        transcriptLines.push(line);
      }
    }

    if (staging.length > 0) {
      transcriptLines.push("[staging]");
      transcriptLines.push(...staging);
    }

    return transcriptLines.join("\n");
  } catch {
    const fallback = fallbackMessages
      .map((message) => `${message.role ?? message.type ?? "message"}: ${extractMessageText(message)}`)
      .filter((line) => !line.endsWith(": "))
      .concat(staging)
      .join("\n");
    return fallback;
  }
}

function normalizeScopeRecommendation(scope: DistillOutput["scope_recommendation"], memoryType: MemoryType): MemoryScope {
  if (scope === "owner_shared") return "owner_shared";
  if (scope === "agent_local") return "agent_local";
  if (memoryType === "summary" || memoryType === "fact" || memoryType === "preference" || memoryType === "decision") {
    return "owner_shared";
  }
  return "agent_local";
}

async function persistDistilledMemories(
  deps: ContextEngineDeps,
  session: SessionState,
  distilled: DistillOutput,
): Promise<{ insertedCount: number; firstMemoryId?: string; insertedTypes: string[] }> {
  const owner = resolveOwnerForSession(deps, session);
  if (!owner) return { insertedCount: 0, insertedTypes: [] };

  const now = Date.now();
  const items: Array<{ type: MemoryType; text: string; scope?: MemoryScope }> = [];

  if (distilled.session_summary) items.push({ type: "summary", text: distilled.session_summary });
  for (const text of distilled.confirmed_facts) items.push({ type: "fact", text });
  for (const text of distilled.decisions) items.push({ type: "decision", text });
  for (const text of distilled.pitfalls) items.push({ type: "pitfall", text });
  for (const text of distilled.preference_updates) items.push({ type: "preference", text });
  for (const text of distilled.environment_truths) items.push({ type: "fact", text });
  for (const text of distilled.open_loops) items.push({ type: "todo", text, scope: distilled.scope_recommendation === "both" ? "agent_local" : undefined });

  let firstMemoryId: string | undefined;
  const insertedTypes: string[] = [];
  let insertedCount = 0;

  for (const item of items) {
    const text = item.text.trim();
    if (!text) continue;
    const memoryScope = item.scope ?? normalizeScopeRecommendation(distilled.scope_recommendation, item.type);
    const record: MemoryRecord = {
      memory_id: randomUUID(),
      owner_namespace: owner.ownerNamespace,
      owner_id: owner.ownerId,
      agent_id: session.agentId ?? "unknown-agent",
      memory_scope: memoryScope,
      memory_type: item.type,
      title: generateTitle(text),
      content: text,
      summary: generateSummary(text),
      tags: JSON.stringify(["distilled", `session:${session.sessionId}`]),
      importance: item.type === "summary" ? 4 : 3,
      confidence: 0.8,
      status: "active",
      supersedes_id: "",
      created_at: now,
      updated_at: now,
      last_used_at: now,
      source_session_id: session.sessionId,
      embedding: await deps.embedder.embed(text),
    };
    await deps.storage.insertMemory(record);
    await deps.storage.insertEvent({
      event_id: randomUUID(),
      memory_id: record.memory_id,
      event_type: "distill",
      event_time: now,
      details_json: JSON.stringify({
        sessionId: session.sessionId,
        scope_recommendation: distilled.scope_recommendation,
        memory_type: item.type,
      }),
    });
    if (!firstMemoryId) firstMemoryId = record.memory_id;
    insertedTypes.push(item.type);
    insertedCount += 1;
  }

  return { insertedCount, firstMemoryId, insertedTypes };
}

export async function compactSession(
  deps: ContextEngineDeps,
  params: {
    sessionId: string;
    sessionFile: string;
    messages?: AgentMessageLike[];
    currentTokenCount?: number;
    customInstructions?: string;
  },
): Promise<CompactResult> {
  const session = upsertSessionState(deps, {
    sessionId: params.sessionId,
    sessionFile: params.sessionFile,
  });

  const transcript = await readTranscriptText(params.sessionFile, params.messages ?? [], session.staging);
  if (!transcript.trim()) {
    return { ok: true, compacted: false, reason: "empty_transcript" };
  }

  const distilled = await deps.distiller.distillTranscript(transcript, {
    customInstructions: params.customInstructions,
  });
  const persisted = await persistDistilledMemories(deps, session, distilled);
  session.staging = [];
  session.updatedAt = Date.now();
  deps.sessionStates.set(session.sessionId, session);

  return {
    ok: true,
    compacted: persisted.insertedCount > 0,
    result: {
      summary: distilled.session_summary,
      firstKeptEntryId: persisted.firstMemoryId,
      tokensBefore: params.currentTokenCount ?? estimateTokens(transcript),
      tokensAfter: Math.max(estimateTokens(distilled.session_summary), 1),
      details: {
        insertedCount: persisted.insertedCount,
        insertedTypes: persisted.insertedTypes,
        scopeRecommendation: distilled.scope_recommendation,
      },
    },
  };
}

export function createMemoryBrainContextEngine(deps: ContextEngineDeps) {
  return {
    info: {
      id: "memory-lancedb-brain",
      name: "Memory LanceDB Brain",
      ownsCompaction: true,
    },

    async bootstrap(params: { sessionId: string; sessionFile: string }) {
      upsertSessionState(deps, { sessionId: params.sessionId, sessionFile: params.sessionFile });
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
      const session = upsertSessionState(deps, {
        sessionId: params.sessionId,
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
      const ownerShared = trimMemoriesByTokenBudget(await selectRelevantMemories(deps, session, "owner_shared", 5), 800);
      const agentLocal = trimMemoriesByTokenBudget(await selectRelevantMemories(deps, session, "agent_local", 3), 400);
      const sections = [
        renderMemorySection("OWNER SHARED MEMORY", ownerShared),
        renderMemorySection("AGENT LOCAL MEMORY", agentLocal),
      ].filter(Boolean);
      const addition = sections.join("\n\n");
      return {
        messages: params.messages,
        estimatedTokens: addition ? estimateTokens(addition) : 0,
        systemPromptAddition: addition || undefined,
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
        owner: parent.owner,
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
    description: "Manual memory operations for memory-lancedb-brain.",
    acceptsArgs: true,
    handler: async (ctx: { args?: string; channel: string }) => {
      const args = (ctx.args ?? "").trim();
      if (!args.toLowerCase().startsWith("distill")) {
        return { text: "Usage: /memory distill" };
      }

      const sessionId = deps.lastSessionByChannel.get(ctx.channel) ?? [...deps.sessionStates.keys()].at(-1);
      if (!sessionId) {
        return { text: "No active session available for distillation yet." };
      }

      const session = deps.sessionStates.get(sessionId);
      if (!session?.sessionFile) {
        return { text: "No session transcript available yet for distillation." };
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
        text: `Distilled session ${sessionId}: inserted ${details.insertedCount ?? 0} memories (${Array.isArray(details.insertedTypes) ? details.insertedTypes.join(", ") : "n/a"}).`,
      };
    },
  };
}
