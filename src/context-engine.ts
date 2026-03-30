/**
 * Context engine implementation for memory-lancedb-brain
 * Phase 5: Auto-distill with session lifecycle hooks
 */

import { readFile, writeFile, readdir } from "node:fs/promises";
import { dirname, basename, join } from "node:path";
import type { MemoryRecord, MemoryEventRecord, MemoryScope, MemoryType } from "./schema.js";
import type { MemoryStorage } from "./storage.js";
import type { OwnerConfig, ResolvedOwner, ResolveOwnerInput } from "./owners.js";
import { resolveOwnerFromContext, normalizeOwners } from "./owners.js";
import type { DistillOutput } from "./distill.js";
import type { LLMCaller } from "./lifecycle.js";
import { buildCandidateMemories, processLifecycle, resolveContradictions, applyConfidenceDecay, consolidateMemories, synthesizeUserProfile } from "./lifecycle.js";
import type { Embedder } from "./embedding.js";
import { importLegacyMarkdown } from "./legacy-import.js";

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
  messageChannel?: string;
  requesterSenderId?: string;
  agentAccountId?: string;
  senderIsOwner?: boolean;
  channelId?: string;
  pendingPrompt?: string;
  pendingPromptContext?: string;
  staging: string[];
  childSessionKeys: string[];
  parentSessionKey?: string;
  subagentEnded?: boolean;
  updatedAt: number;
}

export interface BrainDiagnostics {
  pluginId: string;
  dbPath: string;
  initializedAt: number;
  trustedPluginExplicit?: boolean;
  recentWarnings: Array<{ at: number; message: string }>;
  recentErrors: Array<{ at: number; message: string }>;
  lastBootstrap?: {
    at: number;
    sessionId: string;
    sessionFile?: string;
    agentId?: string;
    ownerId?: string;
    ownerNamespace?: string;
  };
  lastAssemble?: {
    at: number;
    sessionId: string;
    agentId?: string;
    ownerId?: string;
    query: string;
    ownerSharedCount: number;
    agentLocalCount: number;
    skipped?: boolean;
    reason?: string;
  };
  lastAfterTurn?: {
    at: number;
    sessionId: string;
    sessionFile?: string;
    agentId?: string;
    ownerId?: string;
    stagedCount: number;
  };
  lastCompact?: {
    at: number;
    sessionId: string;
    ok: boolean;
    compacted: boolean;
    insertedCount?: number;
    reason?: string;
  };
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
  llmCaller?: LLMCaller;
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
  diagnostics?: BrainDiagnostics;
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

/**
 * Resolve owner from runtimeContext fields or fall back to deps.owners config.
 * This bridges the gap between what openclaw passes to context engine hooks
 * (runtimeContext with messageChannel, senderIsOwner, etc.) and what
 * memory-lancedb-brain needs (ownerId + ownerNamespace).
 */
function resolveOwnerFallback(
  deps: ContextEngineDeps,
  runtimeContext?: Record<string, unknown>,
): ResolvedOwner | undefined {
  // Try resolveOwnerFromContext with runtimeContext fields
  if (runtimeContext) {
    const requesterSenderId =
      typeof runtimeContext.requesterSenderId === "string" ? runtimeContext.requesterSenderId : undefined;
    const agentAccountId =
      typeof runtimeContext.agentAccountId === "string" ? runtimeContext.agentAccountId : undefined;
    const input: ResolveOwnerInput = {
      senderId: requesterSenderId ?? agentAccountId,
      messageChannel: typeof runtimeContext.messageChannel === "string" ? runtimeContext.messageChannel : undefined,
      agentId: typeof runtimeContext.agentId === "string" ? runtimeContext.agentId : undefined,
      senderIsOwner: typeof runtimeContext.senderIsOwner === "boolean" ? runtimeContext.senderIsOwner : undefined,
    };
    const resolved = resolveOwnerFromContext(input, normalizeOwners(deps.owners));
    if (resolved) return resolved;
  }

  // Fall back to first configured owner
  if (deps.owners.length > 0) {
    return {
      ownerId: deps.owners[0].owner_id,
      ownerNamespace: deps.owners[0].owner_namespace,
    };
  }

  return undefined;
}

function pushRecent(list: Array<{ at: number; message: string }> | undefined, message: string, max = 8): void {
  if (!list) return;
  list.push({ at: Date.now(), message });
  if (list.length > max) list.splice(0, list.length - max);
}

function recordWarning(deps: ContextEngineDeps, message: string): void {
  pushRecent(deps.diagnostics?.recentWarnings, message);
  console.warn(`[memory-lancedb-brain] ${message}`);
}

function recordError(deps: ContextEngineDeps, message: string): void {
  pushRecent(deps.diagnostics?.recentErrors, message);
  console.error(`[memory-lancedb-brain] ${message}`);
}

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
    messageChannel?: string;
    requesterSenderId?: string;
    agentAccountId?: string;
    senderIsOwner?: boolean;
    channelId?: string;
    parentSessionKey?: string;
    pendingPrompt?: string;
    pendingPromptContext?: string;
  },
): SessionState {
  const existing = deps.sessionStates.get(params.sessionId);
  if (existing) {
    // Update mutable fields on existing state
    if (params.sessionFile) existing.sessionFile = params.sessionFile;
    if (params.sessionKey) existing.sessionKey = params.sessionKey;
    if (params.agentId) existing.agentId = params.agentId;
    if (params.messageChannel) existing.messageChannel = params.messageChannel;
    if (params.requesterSenderId) existing.requesterSenderId = params.requesterSenderId;
    if (params.agentAccountId) existing.agentAccountId = params.agentAccountId;
    if (typeof params.senderIsOwner === "boolean") existing.senderIsOwner = params.senderIsOwner;
    if (params.channelId) existing.channelId = params.channelId;
    if (typeof params.pendingPrompt === "string") existing.pendingPrompt = params.pendingPrompt;
    if (typeof params.pendingPromptContext === "string") existing.pendingPromptContext = params.pendingPromptContext;
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
    messageChannel: params.messageChannel,
    requesterSenderId: params.requesterSenderId,
    agentAccountId: params.agentAccountId,
    senderIsOwner: params.senderIsOwner,
    channelId: params.channelId,
    pendingPrompt: params.pendingPrompt,
    pendingPromptContext: params.pendingPromptContext,
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
  if (message.role === "tool" || message.type === "toolCall" || message.type === "toolResult") return "";
  if (message.text) return typeof message.text === "string" ? message.text : String(message.text);
  if (message.content) {
    const content = message.content;
    if (Array.isArray(content)) {
      const textBlocks = content
        .filter((c: any) => c?.type === "text" && typeof c?.text === "string")
        .map((c: any) => c.text);
      if (textBlocks.length > 0) return textBlocks.join(" ");
      const onlyToolish = content.every((c: any) =>
        c && typeof c === "object" && ["toolCall", "toolResult", "reasoning", "thinking"].includes(String(c.type ?? "")),
      );
      if (onlyToolish) return "";
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
  if (trimmed.includes("\"toolCall\"") || trimmed.includes("\"toolResult\"")) return false;
  if (isSyntheticBootstrapQueryText(trimmed)) return false;
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

const MEMORY_INTENT_PATTERN = /(remember|recall|previously|last time|yesterday|before|之前|上次|以前|還記得|記得|提到過|說過|昨天)/i;

function shouldSkipRetrieval(query: string): boolean {
  const trimmed = query.trim();
  if (trimmed.length < 4) return true;
  // Force retrieve for memory-intent queries
  if (MEMORY_INTENT_PATTERN.test(trimmed)) return false;
  if (SKIP_RETRIEVAL_PATTERNS.some(p => p.test(trimmed))) return true;
  // Skip very short non-question messages
  const hasCJK = /[\u4e00-\u9fff]/.test(trimmed);
  if (trimmed.length < (hasCJK ? 6 : 15) && !trimmed.includes('?') && !trimmed.includes('？')) return true;
  return false;
}

function isMemoryIntentQuery(query: string): boolean {
  return MEMORY_INTENT_PATTERN.test(query.trim());
}

function isSyntheticBootstrapQueryText(text: string): boolean {
  const trimmed = text.trim();
  return trimmed === "(session bootstrap)" || isSyntheticStartupText(trimmed);
}

function stripConversationMetadata(text: string): string {
  let stripped = text.replace(/\r\n/g, "\n");
  stripped = stripped.replace(/Conversation info \(untrusted metadata\):\s*```json[\s\S]*?```\s*/gi, "");
  stripped = stripped.replace(/Sender \(untrusted metadata\):\s*```json[\s\S]*?```\s*/gi, "");
  stripped = stripped.replace(/^\[[^\]]*GMT[^\]]*\]\s*/gm, "");
  return stripped.trim();
}

function isSyntheticStartupText(text: string): boolean {
  const trimmed = text.trim();
  return trimmed.includes("A new session was started via /new or /reset")
    && trimmed.includes("Execute your Session Startup sequence now");
}

function sanitizeDistillMessageText(text: string): string {
  const stripped = stripConversationMetadata(text);
  if (!stripped) return "";
  if (isSyntheticBootstrapQueryText(stripped)) return "";
  if (stripped.startsWith("/") && !stripped.includes(":")) return "";
  if (stripped.includes("<relevant-memories>") || stripped.includes("[MEMORY_RECALL]")) return "";
  return stripped.trim();
}

function buildDistillTranscript(
  lines: string[],
  tokenBudget: number,
): { transcript: string; estimatedTokens: number; keptMessages: number } {
  const collected: string[] = [];
  let totalTokens = 0;
  let keptMessages = 0;

  for (let i = lines.length - 1; i >= 0; i -= 1) {
    try {
      const entry = JSON.parse(lines[i]);
      if (entry?.type !== "message") continue;

      const role = entry.message?.role;
      const content = entry.message?.content;
      if (role !== "user" && role !== "assistant") continue;

      const rawText = typeof content === "string"
        ? content
        : Array.isArray(content)
          ? content.filter((c: any) => c?.type === "text").map((c: any) => c.text).join(" ")
          : JSON.stringify(content);

      const text = sanitizeDistillMessageText(rawText);
      if (!text || text.length <= 5) continue;

      let formatted = `${role}: ${text}`;
      let lineTokens = Math.ceil(formatted.length / 4);
      if (totalTokens + lineTokens > tokenBudget) {
        const remainingTokens = tokenBudget - totalTokens;
        const overheadTokens = Math.ceil((role.length + 2) / 4);
        const textTokenBudget = remainingTokens - overheadTokens;
        if (textTokenBudget < 8) break;
        const maxChars = Math.max(32, textTokenBudget * 4);
        formatted = `${role}: ${text.slice(0, maxChars).trim()}`;
        lineTokens = Math.ceil(formatted.length / 4);
      }

      collected.unshift(formatted);
      totalTokens += lineTokens;
      keptMessages += 1;
      if (totalTokens >= tokenBudget) break;
    } catch {
      // Ignore parse errors
    }
  }

  return { transcript: collected.join("\n"), estimatedTokens: totalTokens, keptMessages };
}

// Post-retrieval scoring: recency boost + importance weighting + type boosting (item 6)
function applyPostScoring(memories: MemoryRecord[]): MemoryRecord[] {
  if (memories.length <= 1) return memories;
  const now = Date.now();
  const scored = memories.map(m => {
    // Recency boost: half-life 14 days, max +0.15 bonus
    const ageDays = (now - (m.updated_at || m.created_at)) / 86_400_000;
    const recencyBoost = Math.exp(-ageDays / 14) * 0.15;
    // Importance weight: brain uses 1-5 scale, normalize to 0-1
    const importanceWeight = 0.7 + 0.3 * ((m.importance - 1) / 4);
    // Item 3: Confidence factor — decayed memories rank lower
    const confidenceFactor = 0.5 + 0.5 * m.confidence;
    // Item 6: Proactive recall — boost corrections, best_practices, goals
    const typeBoost = (m.memory_type === "correction" || m.memory_type === "best_practice") ? 1.15
      : m.memory_type === "goal" ? 1.10
      : 1.0;
    const score = (1 + recencyBoost) * importanceWeight * confidenceFactor * typeBoost;
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
    // Owner isolation: only return memories belonging to this session's owner
    if (session.owner?.ownerId) {
      filters.owner_id = session.owner.ownerId;
    }
    if (session.owner?.ownerNamespace) {
      filters.owner_namespace = session.owner.ownerNamespace;
    }
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
    recordError(deps, `selectRelevantMemories error (scope=${scope}): ${error instanceof Error ? error.stack : String(error)}`);
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

function buildRecallQuery(messages: AgentMessageLike[], session: SessionState): string {
  const userTurns = messages
    .filter((message) => String(message?.role ?? "").toLowerCase() === "user")
    .map((message) => extractMessageText(message))
    .map((text) => stripConversationMetadata(text))
    .map((text) => text.trim())
    .filter((text) => text.length > 0 && !isSyntheticBootstrapQueryText(text));

  const currentUserQuery = userTurns.at(-1)?.slice(0, 600) ?? "";

  if (currentUserQuery) return currentUserQuery;

  const pendingPromptQuery = stripConversationMetadata(session.pendingPrompt ?? "").trim().slice(0, 600);
  if (pendingPromptQuery && !isSyntheticBootstrapQueryText(pendingPromptQuery)) return pendingPromptQuery;

  return session.staging
    .map((text) => stripConversationMetadata(text))
    .filter((text) => {
      const trimmed = text.trim();
      if (!trimmed) return false;
      if (trimmed.startsWith("[") && trimmed.includes("GMT+")) return false;
      if (trimmed.startsWith("{") && trimmed.includes("\"toolCall\"")) return false;
      if (isSyntheticBootstrapQueryText(trimmed)) return false;
      return true;
    })
    .at(-1)?.slice(0, 600) ?? "";
}

function renderMemorySection(title: string, memories: MemoryRecord[]): string {
  if (memories.length === 0) return "";
  const items = memories.map(m => `- ${m.content}`).join("\n");
  return `### ${title}\n\n${items}`;
}

function buildMemoryContractBlock(params: {
  ownerShared: MemoryRecord[];
  agentLocal: MemoryRecord[];
  memoryIntentQuery: boolean;
  noHitForIntentQuery: boolean;
}): string {
  const sections = [
    renderMemorySection("OWNER SHARED MEMORY", params.ownerShared),
    renderMemorySection("AGENT LOCAL MEMORY", params.agentLocal),
  ].filter(Boolean);

  const lines = [
    "<long-term-memory>",
    "Long-term memory is provided by memory-lancedb-brain and persists across sessions.",
    "Use any retrieved memories naturally. Do not narrate internal memory mechanics unless the user asks.",
  ];

  if (params.noHitForIntentQuery) {
    lines.push("No relevant long-term memory was retrieved for this query.");
    lines.push("If the user asks about yesterday, previous discussions, or remembered facts, say you did not find relevant long-term memory right now.");
    lines.push("Do NOT claim that a restart, /new, or a fresh session wipes all memory or returns you to a blank initial state.");
  } else if (params.memoryIntentQuery) {
    lines.push("The user is explicitly asking about prior discussions or remembered facts.");
    lines.push("Answer directly using the retrieved memories below.");
    lines.push("Do NOT say that you do not remember, that memory was reset, or that every fresh session starts blank while relevant memories are listed below.");
  } else {
    lines.push("Do NOT announce that you remember things or say \"I've noted that\". Just use the knowledge naturally.");
  }

  if (sections.length > 0) {
    lines.push("");
    lines.push(sections.join("\n\n"));
  }

  lines.push("</long-term-memory>");
  return lines.join("\n");
}

function buildPromptContextBlock(params: {
  ownerShared: MemoryRecord[];
  agentLocal: MemoryRecord[];
  memoryIntentQuery: boolean;
  noHitForIntentQuery: boolean;
}): string | undefined {
  if (!params.memoryIntentQuery) return undefined;

  const memoryLines = [...params.ownerShared, ...params.agentLocal].map((memory) => `- ${memory.content}`);
  const lines = [
    "[Long-term memory check for this turn]",
  ];

  if (params.noHitForIntentQuery) {
    lines.push("No relevant long-term memory was found for this query.");
    lines.push("If asked about previous discussions or yesterday, say you did not find relevant long-term memory right now.");
    lines.push("Do not claim that /new, restart, or a fresh session wipes all memory.");
    return lines.join("\n");
  }

  if (memoryLines.length === 0) return undefined;

  lines.push("The user is asking about prior discussions or remembered facts.");
  lines.push("Use the recalled memories below directly if they answer the question.");
  lines.push("Do not say you found no relevant long-term memory while these memories are listed.");
  lines.push("");
  lines.push(...memoryLines);
  return lines.join("\n");
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

    const distillTokenBudget = Math.max(40, deps.autoDistill?.tokenBudget ?? DEFAULT_AUTO_DISTILL.tokenBudget);
    const { transcript: fullTranscript, keptMessages: keptTranscriptMessages } = buildDistillTranscript(lines.slice(-100), distillTokenBudget);

    if (keptTranscriptMessages < 2 || !fullTranscript.trim()) {
      return { ok: true, compacted: false, reason: "Insufficient conversation content" };
    }

    // Distill using LLM
    const distillResult = await deps.distiller.distillTranscript(fullTranscript, {
      customInstructions: params.customInstructions,
    });

    // Fail-closed: refuse to write memories without owner context
    if (!session?.owner?.ownerId || !session?.owner?.ownerNamespace) {
      return { ok: false, compacted: false, reason: "Missing owner context — cannot write memories without owner_id and owner_namespace" };
    }

    const ownerId = session.owner.ownerId;
    const ownerNamespace = session.owner.ownerNamespace;
    const agentId = session?.agentId ?? "unknown";

    // Item 2: Resolve contradictions before inserting new memories
    if ((distillResult.contradictions ?? []).length > 0) {
      const { resolved } = await resolveContradictions(
        { storage: deps.storage, embedder: deps.embedder },
        distillResult.contradictions,
        ownerId,
        ownerNamespace,
      );
      if (resolved > 0) {
        console.log(`[memory-lancedb-brain] compactSession: resolved ${resolved} contradictions`);
      }
    }

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

    // Truncate session file to reduce token count.
    // Keep header lines (non-message) + compaction summary + recent messages.
    const tokenBudget = params.currentTokenCount
      ? Math.floor(params.currentTokenCount * 0.4)  // target 40% of current
      : 40000;
    const headerLines: string[] = [];
    const messageLines: string[] = [];
    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        if (entry?.type === "message") {
          messageLines.push(line);
        } else {
          headerLines.push(line);
        }
      } catch {
        headerLines.push(line);
      }
    }

    // Keep recent messages within token budget
    const keptMessages: string[] = [];
    let keptTokens = 0;
    for (let i = messageLines.length - 1; i >= 0; i--) {
      const lineTokens = Math.ceil(messageLines[i].length / 4);
      if (keptTokens + lineTokens > tokenBudget) break;
      keptMessages.unshift(messageLines[i]);
      keptTokens += lineTokens;
    }

    const droppedCount = messageLines.length - keptMessages.length;
    if (droppedCount > 0) {
      // Chunk and save dropped messages as episodic memory
      const droppedMessages = messageLines.slice(0, droppedCount);
      const chunkSize = 10; // group 10 lines at a time
      
      let episodicChunksInserted = 0;
      for (let i = 0; i < droppedMessages.length; i += chunkSize) {
        const chunkLines = droppedMessages.slice(i, i + chunkSize);
        
        try {
          // Extract text content from JSONL message entries for cleaner embedding
          const chunkText = chunkLines.map(line => {
            try {
              const parsed = JSON.parse(line);
              if (parsed.type !== "message") return "";
              const content = parsed.message?.content;
              const normalized = typeof content === "string"
                ? content
                : Array.isArray(content)
                  ? content.filter((c: any) => c?.type === "text").map((c: any) => c.text).join(" ")
                  : JSON.stringify(content);
              return `${parsed.message?.role || "unknown"}: ${normalized}`;
            } catch {
              return line;
            }
          }).filter(Boolean).join("\n");
          
          if (!chunkText.trim()) continue;
          
          const headerPrefix = `[Session Transcript Chunk ${Math.floor(i/chunkSize) + 1}/${Math.ceil(droppedMessages.length/chunkSize)}]\n`;
          const fullContent = headerPrefix + chunkText;
          
          const vector = await deps.embedder.embed(fullContent);
          
          const record: MemoryRecord = {
            memory_id: `episode-${Date.now()}-${i}`,
            owner_id: ownerId,
            owner_namespace: ownerNamespace,
            agent_id: agentId,
            memory_scope: "session_distilled",
            memory_type: "episode",
            title: `Session Transcript Chunk ${Math.floor(i/chunkSize) + 1}`,
            content: fullContent,
            summary: "",
            tags: "[]",
            importance: 1, // Episode chunks have low intrinsic importance
            confidence: 1.0,
            status: "active",
            supersedes_id: "",
            source: "distill",
            source_session_id: params.sessionId,
            created_at: Date.now(),
            updated_at: Date.now(),
            last_used_at: Date.now(),
            embedding: vector,
          };
          
          await deps.storage.insertMemory(record);
          episodicChunksInserted++;
        } catch (err) {
          console.warn(`[memory-lancedb-brain] Failed to insert episode chunk: ${err instanceof Error ? err.message : String(err)}`);
        }
      }

      // Insert a compaction marker
      const compactionEntry = JSON.stringify({
        type: "compaction",
        id: `compaction-${Date.now()}`,
        timestamp: new Date().toISOString(),
        tokensBefore: params.currentTokenCount ?? Math.ceil(sessionContent.length / 4),
        summary: distillResult.session_summary || "Session compacted by memory-lancedb-brain",
        droppedMessages: droppedCount,
        keptMessages: keptMessages.length,
        episodesSaved: episodicChunksInserted,
      });

      const truncatedContent = [
        ...headerLines,
        compactionEntry,
        ...keptMessages,
      ].join("\n") + "\n";

      await writeFile(resolvedFile, truncatedContent, "utf-8");
      console.log(`[memory-lancedb-brain] compactSession: truncated session file — dropped ${droppedCount} messages (saved as ${episodicChunksInserted} episodes), kept ${keptMessages.length}, ~${keptTokens} tokens`);
    }

    return {
      ok: true,
      compacted: droppedCount > 0,
      result: {
        summary: distillResult.session_summary || "Session distilled",
        insertedCount: lifecycleResult.inserted,
        details: {
          insertedCount: lifecycleResult.inserted,
          merged: lifecycleResult.merged,
          supersedeCandidates: lifecycleResult.supersedeCandidates,
          insertedTypes: lifecycleResult.insertedTypes,
          droppedMessages: droppedCount,
          keptMessages: keptMessages.length,
        },
      },
    };
  } catch (error) {
    recordError(
      deps,
      `compactSession failed for session ${params.sessionId}: ${error instanceof Error ? error.message : String(error)}`,
    );
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
    recordWarning(deps, `Auto-distill skipped (${trigger}): no sessionFile for session ${session.sessionId}`);
    return false;
  }

  if (session.staging.length < config.minStagingLength) {
    recordWarning(
      deps,
      `Auto-distill skipped (${trigger}): staging ${session.staging.length} < min ${config.minStagingLength}`,
    );
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

    recordWarning(deps, `Auto-distill no-op (${trigger}): ${result.reason ?? "no memories produced"}`);
    return false;
  } catch (error) {
    recordError(deps, `Auto-distill failed (${trigger}): ${error instanceof Error ? error.message : String(error)}`);
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
      // Resolve owner from plugin config so session state has it for compaction
      const fallbackOwner = resolveOwnerFallback(deps);
      const session = upsertSessionState(deps, {
        sessionId: params.sessionId,
        sessionKey: params.sessionFile.replace(/\.jsonl$/, ""),
        sessionFile: params.sessionFile,
        ownerId: fallbackOwner?.ownerId,
        ownerNamespace: fallbackOwner?.ownerNamespace,
      });
      if (deps.diagnostics) {
        deps.diagnostics.lastBootstrap = {
          at: Date.now(),
          sessionId: params.sessionId,
          sessionFile: params.sessionFile,
          agentId: session.agentId,
          ownerId: session.owner?.ownerId,
          ownerNamespace: session.owner?.ownerNamespace,
        };
      }
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
      runtimeContext?: Record<string, unknown>;
    }) {
      console.log(`[memory-lancedb-brain] afterTurn: sessionId=${params.sessionId} sessionFile=${params.sessionFile} messages=${params.messages.length}`);
      // Ensure owner is set on session — use runtimeContext or plugin config
      const ownerForSession = resolveOwnerFallback(deps, params.runtimeContext);
      const runtimeAgentId =
        typeof params.runtimeContext?.agentId === "string" ? params.runtimeContext.agentId : undefined;
      const session = upsertSessionState(deps, {
        sessionId: params.sessionId,
        sessionKey: params.sessionFile.replace(/\.jsonl$/, ""),
        sessionFile: params.sessionFile,
        agentId: runtimeAgentId,
        ownerId: ownerForSession?.ownerId,
        ownerNamespace: ownerForSession?.ownerNamespace,
        messageChannel:
          typeof params.runtimeContext?.messageChannel === "string" ? params.runtimeContext.messageChannel : undefined,
        requesterSenderId:
          typeof params.runtimeContext?.requesterSenderId === "string"
            ? params.runtimeContext.requesterSenderId
            : undefined,
        agentAccountId:
          typeof params.runtimeContext?.agentAccountId === "string" ? params.runtimeContext.agentAccountId : undefined,
        senderIsOwner:
          typeof params.runtimeContext?.senderIsOwner === "boolean" ? params.runtimeContext.senderIsOwner : undefined,
        channelId: typeof params.runtimeContext?.channelId === "string" ? params.runtimeContext.channelId : undefined,
      });
      for (const message of params.messages.slice(params.prePromptMessageCount)) {
        const text = extractMessageText(message);
        if (shouldStageText(text)) session.staging.push(text);
      }
      if (params.autoCompactionSummary && shouldStageText(params.autoCompactionSummary)) {
        session.staging.push(params.autoCompactionSummary);
      }
      session.pendingPrompt = undefined;
      session.pendingPromptContext = undefined;
      session.staging = session.staging.slice(-30);
      deps.sessionStates.set(session.sessionId, session);
      if (deps.diagnostics) {
        deps.diagnostics.lastAfterTurn = {
          at: Date.now(),
          sessionId: params.sessionId,
          sessionFile: params.sessionFile,
          agentId: session.agentId,
          ownerId: session.owner?.ownerId,
          stagedCount: session.staging.length,
        };
      }
    },

    async assemble(params: { sessionId: string; messages: AgentMessageLike[]; tokenBudget?: number }): Promise<AssembleResult> {
      const session = upsertSessionState(deps, { sessionId: params.sessionId });
      const recallQuery = buildRecallQuery(params.messages, session);

      // Adaptive: skip retrieval for greetings, commands, short affirmations
      if (!recallQuery || shouldSkipRetrieval(recallQuery)) {
        session.pendingPromptContext = undefined;
        deps.sessionStates.set(session.sessionId, session);
        if (deps.diagnostics) {
          deps.diagnostics.lastAssemble = {
            at: Date.now(),
            sessionId: params.sessionId,
            agentId: session.agentId,
            ownerId: session.owner?.ownerId,
            query: recallQuery,
            ownerSharedCount: 0,
            agentLocalCount: 0,
            skipped: true,
            reason: recallQuery ? "query_skipped" : "empty_query",
          };
        }
        return { messages: params.messages, estimatedTokens: 0 };
      }

      const ownerShared = trimMemoriesByTokenBudget(
        await selectRelevantMemories(deps, session, "owner_shared", 15, recallQuery),
        2000,
      );
      const agentLocal = trimMemoriesByTokenBudget(
        await selectRelevantMemories(deps, session, "agent_local", 5, recallQuery),
        800,
      );
      
      // Implicit Semantic Paging: retrieve recent episodic chunks for this session
      let sessionEpisodes: MemoryRecord[] = [];
      if (session.owner?.ownerId && session.owner?.ownerNamespace) {
        try {
          // Address P1 (Scope by namespace and status) and P2 (Push session filter to storage).
          // NOTE: We deliberately do NOT filter by agent_id here because memory-lancedb-brain 
          // is designed for cross-agent recall — if multiple agents share a session, 
          // they should all have access to the truncated transcript chunks for continuity.
          const rawEpisodes = await deps.storage.queryMemoriesByFilter({
            owner_id: session.owner.ownerId,
            owner_namespace: session.owner.ownerNamespace,
            memory_type: "episode",
            memory_scope: "session_distilled",
            status: "active",
            source_session_id: params.sessionId
          });
          
          // Sort in application code as queryMemoriesByFilter doesn't support sorting yet,
          // but at least we've narrowed down the result set at the database layer.
          const sessionSpecificEpisodes = rawEpisodes
            .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
            .slice(0, 3); // take the 3 most recent chunks
          sessionEpisodes = trimMemoriesByTokenBudget(sessionSpecificEpisodes, 1500); // Allow up to ~1500 tokens of raw history
        } catch (err) {
          console.warn(`[memory-lancedb-brain] Failed to retrieve episode chunks: ${err instanceof Error ? err.message : String(err)}`);
        }
      }

      console.log(`[memory-lancedb-brain] assemble: sessionId=${params.sessionId} query=${recallQuery.slice(0, 60)}... ownerShared=${ownerShared.length} agentLocal=${agentLocal.length} episodes=${sessionEpisodes.length}`);
      if (deps.diagnostics) {
        deps.diagnostics.lastAssemble = {
          at: Date.now(),
          sessionId: params.sessionId,
          agentId: session.agentId,
          ownerId: session.owner?.ownerId,
          query: recallQuery,
          ownerSharedCount: ownerShared.length,
          agentLocalCount: agentLocal.length,
          skipped: false,
        };
      }

      const noHitForIntentQuery = ownerShared.length === 0 && agentLocal.length === 0 && sessionEpisodes.length === 0 && isMemoryIntentQuery(recallQuery);
      const memoryIntentQuery = isMemoryIntentQuery(recallQuery);
      session.pendingPromptContext = buildPromptContextBlock({
        ownerShared,
        agentLocal,
        memoryIntentQuery,
        noHitForIntentQuery,
      });
      deps.sessionStates.set(session.sessionId, session);
      if (ownerShared.length === 0 && agentLocal.length === 0 && sessionEpisodes.length === 0 && !noHitForIntentQuery) {
        return { messages: params.messages, estimatedTokens: 0 };
      }

      // Item 3: Track last_used_at for retrieved memories (fire-and-forget)
      const allRetrieved = [...ownerShared, ...agentLocal]; // Skip tracking for episodes as they are session-bound anyway
      void Promise.all(allRetrieved.map((m) =>
        deps.storage.updateMemory(m.memory_id, { last_used_at: Date.now() }).catch(() => {}),
      ));

      let memoryBlock = buildMemoryContractBlock({
        ownerShared,
        agentLocal,
        memoryIntentQuery,
        noHitForIntentQuery,
      });

      if (sessionEpisodes.length > 0) {
        const episodeText = sessionEpisodes.map(e => e.content).join("\n\n");
        const episodeBlock = `\n<Relevant_Past_Context>\n<!-- Restored exact transcript fragments from earlier in this session -->\n${episodeText}\n</Relevant_Past_Context>`;
        memoryBlock += episodeBlock;
      }

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
      runtimeContext?: Record<string, unknown>;
    }): Promise<CompactResult> {
      // Ensure owner is set before compaction — last chance to resolve
      const session = deps.sessionStates.get(params.sessionId);
      if (session && !session.owner?.ownerId) {
        const resolved = resolveOwnerFallback(deps, params.runtimeContext);
        if (resolved) {
          session.owner = resolved;
          if (typeof params.runtimeContext?.agentId === "string") session.agentId = params.runtimeContext.agentId;
          if (typeof params.runtimeContext?.messageChannel === "string") session.messageChannel = params.runtimeContext.messageChannel;
          if (typeof params.runtimeContext?.requesterSenderId === "string") session.requesterSenderId = params.runtimeContext.requesterSenderId;
          if (typeof params.runtimeContext?.agentAccountId === "string") session.agentAccountId = params.runtimeContext.agentAccountId;
          if (typeof params.runtimeContext?.senderIsOwner === "boolean") session.senderIsOwner = params.runtimeContext.senderIsOwner;
          deps.sessionStates.set(params.sessionId, session);
        }
      }
      const result = await compactSession(deps, {
        sessionId: params.sessionId,
        sessionFile: params.sessionFile,
        currentTokenCount: params.currentTokenCount,
        force: params.force,
        customInstructions: params.customInstructions,
      });
      if (deps.diagnostics) {
        deps.diagnostics.lastCompact = {
          at: Date.now(),
          sessionId: params.sessionId,
          ok: result.ok,
          compacted: result.compacted,
          insertedCount: Number(result.result?.insertedCount ?? 0) || undefined,
          reason: result.reason,
        };
      }
      return result;
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
        messageChannel: parent.messageChannel,
        requesterSenderId: parent.requesterSenderId,
        agentAccountId: parent.agentAccountId,
        senderIsOwner: parent.senderIsOwner,
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
    description: "Manual memory operations for memory-lancedb-brain: distill, recall, list, store, import-legacy.",
    acceptsArgs: true,
    handler: async (ctx: { args?: string; channel: string }) => {
      const args = (ctx.args ?? "").trim();
      if (!args) {
        return {
          text: `Usage:\n  /memory distill — Force distill current session to LanceDB\n  /memory recall [query] — Search memories by query\n  /memory list [scope] — List memories by scope (owner_shared/agent_local/all)\n  /memory store [text] — Store a single memory immediately\n  /memory import-legacy [path] — Import legacy markdown notes into LanceDB\n  /memory migrate-legacy [path] — Alias of import-legacy\n  /memory consolidate — Merge related memory fragments via LLM\n  /memory profile — Synthesize structured user profile from memories\n  /memory decay — Apply confidence decay to unused memories`,
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

            // Fail-closed: refuse to store without owner context
            if (!session.owner?.ownerId || !session.owner?.ownerNamespace) {
              return { text: "Missing owner context — cannot store memory without owner_id and owner_namespace.", isError: true };
            }
            const agentId = session.agentId || "unknown";
            const ownerId = session.owner.ownerId;
            const ownerNamespace = session.owner.ownerNamespace;

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
              source: "manual",
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

        case "import-legacy":
        case "migrate-legacy": {
          const sessionId = deps.lastSessionByChannel.get(ctx.channel) ?? [...deps.sessionStates.keys()].at(-1);
          if (!sessionId) {
            return { text: "No active session available." };
          }

          const session = deps.sessionStates.get(sessionId);
          if (!session) {
            return { text: "Session not found." };
          }

          if (!session.owner?.ownerId || !session.owner?.ownerNamespace) {
            return { text: "Missing owner context — cannot import legacy memory without owner_id and owner_namespace.", isError: true };
          }

          const targetPath = subArgs.trim() || `${process.env.HOME ?? ""}/.openclaw/workspace/memory`;
          try {
            const importResult = await importLegacyMarkdown(
              {
                storage: deps.storage,
                embedder: deps.embedder,
              },
              {
                rootPath: targetPath,
                ownerId: session.owner.ownerId,
                ownerNamespace: session.owner.ownerNamespace,
                agentId: session.agentId ?? "unknown",
                sourceSessionId: sessionId,
              },
            );
            const errorSuffix = importResult.errors.length > 0
              ? `\nErrors (${importResult.errors.length}):\n- ${importResult.errors.slice(0, 5).join("\n- ")}`
              : "";
            return {
              text: `Legacy import complete.\nDiscovered: ${importResult.filesDiscovered}\nConsidered: ${importResult.filesConsidered}\nImported: ${importResult.imported}\nUpdated: ${importResult.updated}\nSkipped: ${importResult.skipped}${errorSuffix}`,
            };
          } catch (error) {
            return { text: `Legacy import failed: ${error instanceof Error ? error.message : String(error)}`, isError: true };
          }
        }

        case "consolidate": {
          // Item 1: Memory Consolidation
          if (!deps.llmCaller) {
            return { text: "Consolidation unavailable: no LLM caller configured." };
          }

          const sessionId5 = deps.lastSessionByChannel.get(ctx.channel) ?? [...deps.sessionStates.keys()].at(-1);
          const session5 = sessionId5 ? deps.sessionStates.get(sessionId5) : undefined;
          if (!session5?.owner?.ownerId || !session5?.owner?.ownerNamespace) {
            return { text: "Missing owner context — cannot consolidate without owner_id and owner_namespace." };
          }
          const ownerId5 = session5.owner.ownerId;
          const ownerNamespace5 = session5.owner.ownerNamespace;

          try {
            const result = await consolidateMemories(
              { storage: deps.storage, embedder: deps.embedder },
              deps.llmCaller,
              ownerId5,
              ownerNamespace5,
            );
            return {
              text: result.groupsMerged > 0
                ? `Consolidated ${result.consolidated} fragments into ${result.groupsMerged} groups.`
                : "No memory clusters found for consolidation (need ≥3 related memories).",
            };
          } catch (error) {
            return { text: `Consolidation failed: ${error instanceof Error ? error.message : String(error)}`, isError: true };
          }
        }

        case "profile": {
          // Item 4: User Profile Synthesis
          if (!deps.llmCaller) {
            return { text: "Profile synthesis unavailable: no LLM caller configured." };
          }

          const sessionId6 = deps.lastSessionByChannel.get(ctx.channel) ?? [...deps.sessionStates.keys()].at(-1);
          const session6 = sessionId6 ? deps.sessionStates.get(sessionId6) : undefined;
          if (!session6?.owner?.ownerId || !session6?.owner?.ownerNamespace) {
            return { text: "Missing owner context — cannot synthesize profile without owner_id and owner_namespace." };
          }
          const ownerId6 = session6.owner.ownerId;
          const ownerNamespace6 = session6.owner.ownerNamespace;

          try {
            const result = await synthesizeUserProfile(
              { storage: deps.storage, embedder: deps.embedder },
              deps.llmCaller,
              ownerId6,
              ownerNamespace6,
            );
            if (!result) {
              return { text: "Not enough memories to synthesize a profile (need ≥3)." };
            }
            return { text: `User profile synthesized:\n\n${result.profile}\n\nStored as memory ${result.profileMemoryId.slice(0, 8)}...` };
          } catch (error) {
            return { text: `Profile synthesis failed: ${error instanceof Error ? error.message : String(error)}`, isError: true };
          }
        }

        case "decay": {
          // Item 3: Manual confidence decay trigger
          try {
            const result = await applyConfidenceDecay(
              { storage: deps.storage, embedder: deps.embedder },
            );
            return { text: `Confidence decay applied: ${result.decayed} memories updated.` };
          } catch (error) {
            return { text: `Decay failed: ${error instanceof Error ? error.message : String(error)}`, isError: true };
          }
        }

        default:
          return {
            text: `Unknown command: ${command}.\n\nUsage:\n  /memory distill\n  /memory recall [query]\n  /memory list [scope]\n  /memory store [text]\n  /memory import-legacy [path]\n  /memory migrate-legacy [path]\n  /memory consolidate\n  /memory profile\n  /memory decay`,
          };
      }
    },
  };
}
