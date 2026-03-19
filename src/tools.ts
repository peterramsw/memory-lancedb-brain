/**
 * Phase 2 memory tools for memory-lancedb-brain
 */

import { randomUUID } from "node:crypto";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type { MemoryRecord, MemoryScope, MemoryType } from "./schema.js";
import type { MemoryFilters, MemoryStorage } from "./storage.js";
import type { Embedder } from "./embedding.js";
import {
  detectCategory,
  generateSummary,
  generateTitle,
  hybridRetrieve,
  mapCategory,
} from "./retrieval.js";
import {
  getAccessibleScopes,
  normalizeOwners,
  resolveOwnerFromContext,
  validateAccess,
  type OwnerConfig,
} from "./owners.js";

export interface ToolCoreContext {
  storage: MemoryStorage;
  embedder: Embedder;
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
    rerankModel?: string;
    rerankEndpoint?: string;
    candidatePoolSize?: number;
  };
}

interface ToolRuntimeContext {
  agentId?: string;
  sessionId?: string;
  sessionKey?: string;
  messageChannel?: string;
  requesterSenderId?: string;
  senderIsOwner?: boolean;
}

function content(text: string, details?: Record<string, unknown>) {
  return {
    content: [{ type: "text", text }],
    ...(details ? { details } : {}),
  };
}

function parseScope(scope?: unknown): MemoryScope {
  if (scope === "owner_shared" || scope === "agent_local" || scope === "session_distilled") {
    return scope;
  }
  return "agent_local";
}

function sanitizeMemory(memory: MemoryRecord) {
  const { embedding, ...rest } = memory;
  return rest;
}

function toImportance(raw: unknown, fallback = 3): number {
  const value = Number(raw);
  if (!Number.isFinite(value)) return fallback;
  return Math.max(1, Math.min(5, Math.round(value)));
}

function resolveMemoryType(rawCategory: unknown, text: string): MemoryType {
  return mapCategory(typeof rawCategory === "string" ? rawCategory : undefined) ?? detectCategory(text);
}

function buildBaseFilters(
  owner: { ownerId: string; ownerNamespace: string },
  agentId: string | undefined,
  scope?: MemoryScope,
  category?: MemoryType,
): MemoryFilters {
  const filters: MemoryFilters = {
    owner_id: owner.ownerId,
    owner_namespace: owner.ownerNamespace,
  };

  if (scope) filters.memory_scope = scope;
  if (category) filters.memory_type = category;
  if (scope === "agent_local" || scope === "session_distilled") {
    filters.agent_id = agentId;
  }
  return filters;
}

async function resolveExecutionContext(core: ToolCoreContext, runtime: ToolRuntimeContext) {
  const owners = normalizeOwners(core.owners);
  const owner = resolveOwnerFromContext(
    {
      senderId: runtime.requesterSenderId,
      messageChannel: runtime.messageChannel,
      agentId: runtime.agentId,
      senderIsOwner: runtime.senderIsOwner,
    },
    owners,
  );

  if (!owner) {
    throw new Error("unable to resolve owner from tool context");
  }

  return {
    owner,
    agentId: runtime.agentId,
    sessionId: runtime.sessionId ?? runtime.sessionKey ?? "unknown-session",
    accessibleScopes: getAccessibleScopes(runtime.agentId, core.agentWhitelist),
  };
}

async function recordEvent(
  storage: MemoryStorage,
  memory_id: string,
  event_type: MemoryRecord["memory_type"] | "create" | "merge" | "promote" | "archive" | "supersede" | "recall" | "distill",
  details: Record<string, unknown>,
) {
  await storage.insertEvent({
    event_id: randomUUID(),
    memory_id,
    event_type: event_type as any,
    event_time: Date.now(),
    details_json: JSON.stringify(details),
  });
}

function createMemoryRecallTool(core: ToolCoreContext, runtime: ToolRuntimeContext) {
  return {
    name: "memory_recall",
    label: "Memory Recall",
    description: "Search through long-term memories using hybrid retrieval.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string" },
        limit: { type: "number" },
        scope: { type: "string" },
        category: { type: "string" },
      },
      required: ["query"],
    },
    async execute(_toolCallId: string, params: Record<string, unknown>) {
      const ctx = await resolveExecutionContext(core, runtime);
      const query = String(params.query ?? "").trim();
      const limit = Math.max(1, Math.min(20, Number(params.limit ?? 5)));
      const scope = params.scope ? parseScope(params.scope) : undefined;
      const category = mapCategory(typeof params.category === "string" ? params.category : undefined);

      if (!query) return content("memory_recall failed: query is required", { error: "missing_query" });
      if (scope) {
        const access = validateAccess(ctx.agentId, scope, core.agentWhitelist);
        if (!access.allowed) return content(`Access denied: ${access.reason}`, { error: "access_denied", scope });
      }

      const results = await hybridRetrieve(core.storage, core.embedder, {
        query,
        ownerId: ctx.owner.ownerId,
        ownerNamespace: ctx.owner.ownerNamespace,
        agentId: ctx.agentId,
        scope,
        category,
        limit,
        mode: core.retrieval?.mode ?? "hybrid",
        vectorWeight: core.retrieval?.vectorWeight,
        bm25Weight: core.retrieval?.bm25Weight,
        minScore: core.retrieval?.minScore,
        hardMinScore: core.retrieval?.hardMinScore,
        rerank: core.retrieval?.rerank,
        rerankApiKey: core.retrieval?.rerankApiKey,
        rerankEndpoint: core.retrieval?.rerankEndpoint,
        rerankModel: core.retrieval?.rerankModel,
        candidatePoolSize: core.retrieval?.candidatePoolSize,
      });

      for (const result of results) {
        await core.storage.updateMemory(result.memory.memory_id, { last_used_at: Date.now() });
        await recordEvent(core.storage, result.memory.memory_id, "recall", {
          query,
          score: result.score,
          reasons: result.reasons,
        });
      }

      return content(
        results.length === 0
          ? "No relevant memories found."
          : `Found ${results.length} memories.`,
        {
          count: results.length,
          memories: results.map((result) => ({
            ...sanitizeMemory(result.memory),
            score: result.score,
            reasons: result.reasons,
          })),
        },
      );
    },
  };
}

function createMemoryStoreTool(core: ToolCoreContext, runtime: ToolRuntimeContext) {
  return {
    name: "memory_store",
    label: "Memory Store",
    description: "Save important information in long-term memory.",
    parameters: {
      type: "object",
      properties: {
        text: { type: "string" },
        importance: { type: "number" },
        category: { type: "string" },
        scope: { type: "string" },
      },
      required: ["text"],
    },
    async execute(_toolCallId: string, params: Record<string, unknown>) {
      const ctx = await resolveExecutionContext(core, runtime);
      const text = String(params.text ?? "").trim();
      const scope = parseScope(params.scope);
      const access = validateAccess(ctx.agentId, scope, core.agentWhitelist);
      if (!access.allowed) return content(`Access denied: ${access.reason}`, { error: "access_denied", scope });
      if (!text) return content("memory_store failed: text is required", { error: "missing_text" });

      const embedding = await core.embedder.embed(text);
      const memoryType = resolveMemoryType(params.category, text);
      const now = Date.now();
      const record: MemoryRecord = {
        memory_id: randomUUID(),
        owner_namespace: ctx.owner.ownerNamespace,
        owner_id: ctx.owner.ownerId,
        agent_id: ctx.agentId ?? "unknown-agent",
        memory_scope: scope,
        memory_type: memoryType,
        title: generateTitle(text),
        content: text,
        summary: generateSummary(text),
        tags: "[]",
        importance: toImportance(params.importance),
        confidence: 0.8,
        status: "active",
        supersedes_id: "",
        created_at: now,
        updated_at: now,
        last_used_at: now,
        source_session_id: ctx.sessionId,
        source: "manual",
        embedding,
      };

      await core.storage.insertMemory(record);
      await recordEvent(core.storage, record.memory_id, "create", {
        scope,
        memory_type: memoryType,
        source_session_id: ctx.sessionId,
      });

      return content(`Stored memory in ${scope}.`, {
        memory_id: record.memory_id,
        scope,
        type: memoryType,
        memory: sanitizeMemory(record),
      });
    },
  };
}

function createMemoryUpdateTool(core: ToolCoreContext, runtime: ToolRuntimeContext) {
  return {
    name: "memory_update",
    label: "Memory Update",
    description: "Update an existing memory in-place.",
    parameters: {
      type: "object",
      properties: {
        memoryId: { type: "string" },
        text: { type: "string" },
        importance: { type: "number" },
        category: { type: "string" },
      },
      required: ["memoryId"],
    },
    async execute(_toolCallId: string, params: Record<string, unknown>) {
      const ctx = await resolveExecutionContext(core, runtime);
      const memoryId = String(params.memoryId ?? "").trim();
      if (!memoryId) return content("memory_update failed: memoryId is required", { error: "missing_memory_id" });

      const rows = await core.storage.queryMemoriesByFilter({
        memory_id: memoryId,
        owner_id: ctx.owner.ownerId,
        owner_namespace: ctx.owner.ownerNamespace,
      });
      const target = rows[0];
      if (!target) return content("Memory not found.", { error: "not_found", memoryId });

      const access = validateAccess(ctx.agentId, target.memory_scope, core.agentWhitelist);
      if (!access.allowed) return content(`Access denied: ${access.reason}`, { error: "access_denied" });
      if ((target.memory_scope === "agent_local" || target.memory_scope === "session_distilled") && target.agent_id !== ctx.agentId) {
        return content("Access denied: memory belongs to another agent", { error: "access_denied" });
      }

      const updates: Partial<MemoryRecord> = {};
      const text = typeof params.text === "string" ? params.text.trim() : undefined;
      if (text) {
        updates.content = text;
        updates.title = generateTitle(text);
        updates.summary = generateSummary(text);
        updates.embedding = await core.embedder.embed(text);
      }
      if (typeof params.importance !== "undefined") updates.importance = toImportance(params.importance, target.importance);
      if (typeof params.category === "string") updates.memory_type = resolveMemoryType(params.category, text ?? target.content);

      await core.storage.updateMemory(memoryId, updates);
      const updated = (await core.storage.queryMemoriesByFilter({ memory_id: memoryId }))[0];
      return content("Memory updated.", {
        memory: updated ? sanitizeMemory(updated) : { memory_id: memoryId },
      });
    },
  };
}

function createMemoryForgetTool(core: ToolCoreContext, runtime: ToolRuntimeContext) {
  return {
    name: "memory_forget",
    label: "Memory Forget",
    description: "Delete a memory by id or query.",
    parameters: {
      type: "object",
      properties: {
        memoryId: { type: "string" },
        query: { type: "string" },
      },
    },
    async execute(_toolCallId: string, params: Record<string, unknown>) {
      const ctx = await resolveExecutionContext(core, runtime);
      let memoryId = typeof params.memoryId === "string" ? params.memoryId.trim() : "";

      if (!memoryId && typeof params.query === "string" && params.query.trim()) {
        const found = await hybridRetrieve(core.storage, core.embedder, {
          query: params.query.trim(),
          ownerId: ctx.owner.ownerId,
          ownerNamespace: ctx.owner.ownerNamespace,
          agentId: ctx.agentId,
          limit: 1,
          mode: core.retrieval?.mode ?? "hybrid",
          vectorWeight: core.retrieval?.vectorWeight,
          bm25Weight: core.retrieval?.bm25Weight,
          minScore: core.retrieval?.minScore,
          hardMinScore: core.retrieval?.hardMinScore,
        });
        memoryId = found[0]?.memory.memory_id ?? "";
      }

      if (!memoryId) return content("memory_forget failed: memoryId or resolvable query is required", { error: "missing_target" });

      const rows = await core.storage.queryMemoriesByFilter({
        memory_id: memoryId,
        owner_id: ctx.owner.ownerId,
        owner_namespace: ctx.owner.ownerNamespace,
      });
      const target = rows[0];
      if (!target) return content("Memory not found.", { error: "not_found", memoryId });

      const access = validateAccess(ctx.agentId, target.memory_scope, core.agentWhitelist);
      if (!access.allowed) return content(`Access denied: ${access.reason}`, { error: "access_denied" });
      if ((target.memory_scope === "agent_local" || target.memory_scope === "session_distilled") && target.agent_id !== ctx.agentId) {
        return content("Access denied: memory belongs to another agent", { error: "access_denied" });
      }

      await core.storage.deleteMemory(memoryId);
      return content("Memory deleted.", { memory_id: memoryId });
    },
  };
}

function createMemoryListTool(core: ToolCoreContext, runtime: ToolRuntimeContext) {
  return {
    name: "memory_list",
    label: "Memory List",
    description: "List recent memories with optional filtering.",
    parameters: {
      type: "object",
      properties: {
        limit: { type: "number" },
        offset: { type: "number" },
        scope: { type: "string" },
        category: { type: "string" },
      },
    },
    async execute(_toolCallId: string, params: Record<string, unknown>) {
      const ctx = await resolveExecutionContext(core, runtime);
      const limit = Math.max(1, Math.min(50, Number(params.limit ?? 10)));
      const offset = Math.max(0, Number(params.offset ?? 0));
      const scope = params.scope ? parseScope(params.scope) : undefined;
      const category = mapCategory(typeof params.category === "string" ? params.category : undefined);

      if (scope) {
        const access = validateAccess(ctx.agentId, scope, core.agentWhitelist);
        if (!access.allowed) return content(`Access denied: ${access.reason}`, { error: "access_denied", scope });
      }

      const rows = await core.storage.queryMemoriesByFilter(buildBaseFilters(ctx.owner, ctx.agentId, scope, category));
      const sorted = rows.sort((a, b) => b.updated_at - a.updated_at).slice(offset, offset + limit);
      return content(`Listed ${sorted.length} memories.`, {
        count: sorted.length,
        total: rows.length,
        memories: sorted.map(sanitizeMemory),
      });
    },
  };
}

function createMemoryStatsTool(core: ToolCoreContext, runtime: ToolRuntimeContext) {
  return {
    name: "memory_stats",
    label: "Memory Stats",
    description: "Get statistics about stored memories.",
    parameters: {
      type: "object",
      properties: {
        scope: { type: "string" },
      },
    },
    async execute(_toolCallId: string, params: Record<string, unknown>) {
      const ctx = await resolveExecutionContext(core, runtime);
      const requestedScope = params.scope ? parseScope(params.scope) : undefined;
      if (requestedScope) {
        const access = validateAccess(ctx.agentId, requestedScope, core.agentWhitelist);
        if (!access.allowed) return content(`Access denied: ${access.reason}`, { error: "access_denied", scope: requestedScope });
      }

      const scopes = requestedScope ? [requestedScope] : ctx.accessibleScopes;
      const rows: MemoryRecord[] = [];
      for (const scope of scopes) {
        rows.push(...(await core.storage.queryMemoriesByFilter(buildBaseFilters(ctx.owner, ctx.agentId, scope))));
      }

      const scopeCounts: Record<string, number> = {};
      const typeCounts: Record<string, number> = {};
      const statusCounts: Record<string, number> = {};
      for (const row of rows) {
        scopeCounts[row.memory_scope] = (scopeCounts[row.memory_scope] ?? 0) + 1;
        typeCounts[row.memory_type] = (typeCounts[row.memory_type] ?? 0) + 1;
        statusCounts[row.status] = (statusCounts[row.status] ?? 0) + 1;
      }

      return content("Memory stats generated.", {
        total: rows.length,
        byScope: scopeCounts,
        byType: typeCounts,
        byStatus: statusCounts,
      });
    },
  };
}

export function registerAllMemoryTools(api: OpenClawPluginApi & { registerTool?: Function }, core: ToolCoreContext): void {
  const tools = [
    { name: "memory_recall", factory: createMemoryRecallTool },
    { name: "memory_store", factory: createMemoryStoreTool },
    { name: "memory_forget", factory: createMemoryForgetTool },
    { name: "memory_update", factory: createMemoryUpdateTool },
    { name: "memory_list", factory: createMemoryListTool },
    { name: "memory_stats", factory: createMemoryStatsTool },
  ];

  for (const tool of tools) {
    api.registerTool?.(
      (runtimeCtx: ToolRuntimeContext) => tool.factory(core, runtimeCtx),
      { name: tool.name },
    );
  }
}
