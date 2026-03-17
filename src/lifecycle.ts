/**
 * Lifecycle management for memory-lancedb-brain
 * Handles merge, supersede detection, and archive operations
 */

import { randomUUID } from "node:crypto";
import type { MemoryEventRecord, MemoryRecord, MemoryScope, MemoryType } from "./schema.js";
import type { MemoryStorage } from "./storage.js";
import type { Embedder } from "./embedding.js";
import { generateSummary, generateTitle } from "./retrieval.js";
import type { DistillOutput } from "./distill.js";

export interface LifecycleDeps {
  storage: MemoryStorage;
  embedder: Embedder;
}

export interface LifecycleSessionInfo {
  sessionId: string;
  ownerId: string;
  ownerNamespace: string;
  agentId?: string;
}

export interface MergeConfig {
  threshold?: number;
}

export interface SupersedeConfig {
  similarityThreshold?: number;
}

export interface ArchiveConfig {
  maxImportance?: number;
  daysInactive?: number;
}

export interface LifecycleProcessResult {
  inserted: number;
  merged: number;
  supersedeCandidates: number;
  archived: number;
  firstMemoryId?: string;
  insertedTypes: string[];
}

const DEFAULT_MERGE_THRESHOLD = 0.92;
const DEFAULT_SUPERSEDE_THRESHOLD = 0.8;
const DEFAULT_MAX_IMPORTANCE = 1;
const DEFAULT_DAYS_INACTIVE = 90;

function cosineSimilarity(a: number[], b: number[]): number {
  if (!a.length || !b.length || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const av = Number(a[i] ?? 0);
    const bv = Number(b[i] ?? 0);
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function normalizeScopeRecommendation(scope: DistillOutput["scope_recommendation"], memoryType: MemoryType): MemoryScope {
  if (scope === "owner_shared") return "owner_shared";
  if (scope === "agent_local") return "agent_local";
  if (["summary", "fact", "preference", "decision"].includes(memoryType)) return "owner_shared";
  return "agent_local";
}

export async function buildCandidateMemories(
  deps: LifecycleDeps,
  session: LifecycleSessionInfo,
  distilled: DistillOutput,
  now = Date.now(),
): Promise<MemoryRecord[]> {
  const items: Array<{ type: MemoryType; text: string; scope?: MemoryScope }> = [];

  if (distilled.session_summary) items.push({ type: "summary", text: distilled.session_summary });
  for (const text of distilled.confirmed_facts) items.push({ type: "fact", text });
  for (const text of distilled.decisions) items.push({ type: "decision", text });
  for (const text of distilled.pitfalls) items.push({ type: "pitfall", text });
  for (const text of distilled.preference_updates) items.push({ type: "preference", text });
  for (const text of distilled.environment_truths) items.push({ type: "fact", text });
  for (const text of distilled.open_loops) {
    items.push({
      type: "todo",
      text,
      scope: distilled.scope_recommendation === "both" ? "agent_local" : undefined,
    });
  }
  for (const text of (distilled.corrections ?? [])) items.push({ type: "correction", text });
  for (const text of (distilled.best_practices ?? [])) items.push({ type: "best_practice", text });

  const candidates: MemoryRecord[] = [];
  for (const item of items) {
    const text = item.text.trim();
    if (!text) continue;
    const embedding = await deps.embedder.embed(text);
    candidates.push({
      memory_id: randomUUID(),
      owner_namespace: session.ownerNamespace,
      owner_id: session.ownerId,
      agent_id: session.agentId ?? "unknown-agent",
      memory_scope: item.scope ?? normalizeScopeRecommendation(distilled.scope_recommendation, item.type),
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
      embedding,
    });
  }

  return candidates;
}

export async function checkMergeCandidate(
  deps: LifecycleDeps,
  candidate: MemoryRecord,
  config: MergeConfig = {},
): Promise<{ existing: MemoryRecord; similarity: number } | null> {
  const threshold = config.threshold ?? DEFAULT_MERGE_THRESHOLD;
  const existing = await deps.storage.queryMemoriesByFilter({
    owner_id: candidate.owner_id,
    owner_namespace: candidate.owner_namespace,
    memory_scope: candidate.memory_scope,
    status: "active",
  });

  let best: { existing: MemoryRecord; similarity: number } | null = null;
  for (const memory of existing) {
    if (memory.memory_id === candidate.memory_id) continue;
    const similarity = cosineSimilarity(memory.embedding, candidate.embedding);
    if (similarity > threshold && (!best || similarity > best.similarity)) {
      best = { existing: memory, similarity };
    }
  }
  return best;
}

export async function mergeOrInsertMemory(
  deps: LifecycleDeps,
  candidate: MemoryRecord,
  config: MergeConfig = {},
): Promise<{ action: "merged" | "inserted"; memoryId: string; similarity?: number }> {
  const mergeMatch = await checkMergeCandidate(deps, candidate, config);

  if (mergeMatch) {
    const kept = mergeMatch.existing;
    const mergedContent = kept.content.length >= candidate.content.length
      ? kept.content
      : `${kept.content}\n\n[merged]\n${candidate.content}`;

    // Bump importance on merge: recurrence = confidence signal (capped at 5)
    const bumpedImportance = Math.min(5, Math.max(kept.importance, candidate.importance) + 1);
    await deps.storage.updateMemory(kept.memory_id, {
      content: mergedContent,
      summary: candidate.summary || kept.summary,
      importance: bumpedImportance,
      confidence: Math.min(1, Math.max(kept.confidence, candidate.confidence) + 0.05),
      last_used_at: Date.now(),
      updated_at: Date.now(),
    });

    await deps.storage.insertEvent({
      event_id: randomUUID(),
      memory_id: kept.memory_id,
      event_type: "merge",
      event_time: Date.now(),
      details_json: JSON.stringify({
        kept_memory_id: kept.memory_id,
        new_candidate_id: candidate.memory_id,
        existing_summary: kept.summary,
        new_summary: candidate.summary,
        new_text: candidate.content.slice(0, 500),
        source_session_id: candidate.source_session_id,
        similarity: mergeMatch.similarity,
      }),
    });

    return { action: "merged", memoryId: kept.memory_id, similarity: mergeMatch.similarity };
  }

  await deps.storage.insertMemory(candidate);
  await deps.storage.insertEvent({
    event_id: randomUUID(),
    memory_id: candidate.memory_id,
    event_type: "create",
    event_time: Date.now(),
    details_json: JSON.stringify({
      source_session_id: candidate.source_session_id,
      memory_type: candidate.memory_type,
      scope: candidate.memory_scope,
    }),
  });
  return { action: "inserted", memoryId: candidate.memory_id };
}

export async function findSupersedeCandidates(
  deps: LifecycleDeps,
  candidate: MemoryRecord,
  config: SupersedeConfig = {},
): Promise<Array<{ existingMemory: MemoryRecord; newCandidate: MemoryRecord; similarity: number }>> {
  const threshold = config.similarityThreshold ?? DEFAULT_SUPERSEDE_THRESHOLD;
  const existing = await deps.storage.queryMemoriesByFilter({
    owner_id: candidate.owner_id,
    owner_namespace: candidate.owner_namespace,
    memory_scope: candidate.memory_scope,
    memory_type: candidate.memory_type,
    status: "active",
  });

  const results: Array<{ existingMemory: MemoryRecord; newCandidate: MemoryRecord; similarity: number }> = [];
  for (const memory of existing) {
    if (memory.memory_id === candidate.memory_id) continue;
    const similarity = cosineSimilarity(memory.embedding, candidate.embedding);
    if (similarity > threshold && similarity <= DEFAULT_MERGE_THRESHOLD && memory.updated_at < candidate.updated_at) {
      results.push({ existingMemory: memory, newCandidate: candidate, similarity });
    }
  }
  return results;
}

export async function recordSupersedeEvents(
  deps: LifecycleDeps,
  candidates: Array<{ existingMemory: MemoryRecord; newCandidate: MemoryRecord; similarity: number }>,
): Promise<void> {
  for (const c of candidates) {
    await deps.storage.insertEvent({
      event_id: randomUUID(),
      memory_id: c.existingMemory.memory_id,
      event_type: "supersede",
      event_time: Date.now(),
      details_json: JSON.stringify({
        candidate_memory_id: c.existingMemory.memory_id,
        candidate_text: c.existingMemory.content.slice(0, 500),
        new_text: c.newCandidate.content.slice(0, 500),
        similarity: c.similarity,
        requires_manual_confirmation: true,
      }),
    });
  }
}

export async function archiveColdMemories(
  deps: LifecycleDeps,
  config: ArchiveConfig = {},
  now = Date.now(),
): Promise<{ archivedCount: number; events: MemoryEventRecord[] }> {
  const maxImportance = config.maxImportance ?? DEFAULT_MAX_IMPORTANCE;
  const daysInactive = config.daysInactive ?? DEFAULT_DAYS_INACTIVE;
  const cutoffTime = now - daysInactive * 24 * 60 * 60 * 1000;

  const active = await deps.storage.queryMemoriesByFilter({ status: "active" });
  const candidates = active.filter((memory) => memory.importance <= maxImportance && memory.last_used_at < cutoffTime);

  const events: MemoryEventRecord[] = [];
  for (const memory of candidates) {
    await deps.storage.updateMemory(memory.memory_id, {
      status: "archived",
      updated_at: now,
    });
    const event: MemoryEventRecord = {
      event_id: randomUUID(),
      memory_id: memory.memory_id,
      event_type: "archive",
      event_time: now,
      details_json: JSON.stringify({
        reason: "cold_memory",
        importance: memory.importance,
        last_used_at: memory.last_used_at,
        days_inactive: Math.floor((now - memory.last_used_at) / (1000 * 60 * 60 * 24)),
      }),
    };
    events.push(event);
    await deps.storage.insertEvent(event);
  }

  return { archivedCount: candidates.length, events };
}

export async function processLifecycle(
  deps: LifecycleDeps,
  candidates: MemoryRecord[],
  mergeConfig: MergeConfig = {},
  supersedeConfig: SupersedeConfig = {},
  archiveConfig?: ArchiveConfig,
): Promise<LifecycleProcessResult> {
  let inserted = 0;
  let merged = 0;
  let supersedeCandidates = 0;
  let firstMemoryId: string | undefined;
  const insertedTypes: string[] = [];

  for (const candidate of candidates) {
    const result = await mergeOrInsertMemory(deps, candidate, mergeConfig);
    if (!firstMemoryId) firstMemoryId = result.memoryId;
    insertedTypes.push(candidate.memory_type);

    if (result.action === "inserted") inserted += 1;
    else merged += 1;

    const supersede = await findSupersedeCandidates(deps, candidate, supersedeConfig);
    supersedeCandidates += supersede.length;
    if (supersede.length > 0) {
      await recordSupersedeEvents(deps, supersede);
    }
  }

  const archive = archiveConfig ? await archiveColdMemories(deps, archiveConfig) : { archivedCount: 0, events: [] };

  return {
    inserted,
    merged,
    supersedeCandidates,
    archived: archive.archivedCount,
    firstMemoryId,
    insertedTypes,
  };
}
