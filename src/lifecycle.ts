/**
 * Lifecycle management for memory-lancedb-brain
 * Handles merge, supersede, archive, consolidation, contradiction, decay, profile
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
  superseded: number;
  supersedeCandidates: number;
  archived: number;
  firstMemoryId?: string;
  insertedTypes: string[];
}

/** LLM caller function type — provided by distill.ts createLLMCaller() */
export type LLMCaller = (systemPrompt: string, userPrompt: string) => Promise<string>;

const DEFAULT_MERGE_THRESHOLD = 0.92;
const DEFAULT_SUPERSEDE_THRESHOLD = 0.8;
const DEFAULT_MAX_IMPORTANCE = 1;
const DEFAULT_DAYS_INACTIVE = 90;

export function cosineSimilarity(a: number[], b: number[]): number {
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
  if (["summary", "fact", "preference", "decision", "goal"].includes(memoryType)) return "owner_shared";
  return "agent_local";
}

export async function buildCandidateMemories(
  deps: LifecycleDeps,
  session: LifecycleSessionInfo,
  distilled: DistillOutput,
  now = Date.now(),
): Promise<MemoryRecord[]> {
  const items: Array<{ type: MemoryType; text: string; scope?: MemoryScope; importance?: number }> = [];

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

  // Item 5: Style observations → preference type
  for (const text of (distilled.style_observations ?? [])) {
    items.push({ type: "preference", text, scope: "owner_shared" });
  }

  // Item 8: Expertise signals → fact type
  for (const text of (distilled.expertise_signals ?? [])) {
    items.push({ type: "fact", text, scope: "owner_shared" });
  }

  // Item 9: Active goals → goal type
  for (const text of (distilled.active_goals ?? [])) {
    items.push({ type: "goal", text, scope: "owner_shared", importance: 4 });
  }

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
      importance: item.importance ?? (item.type === "summary" ? 4 : 3),
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

/**
 * Item 2: Contradiction Detection & Resolution
 * Process contradictions from distiller output — find and supersede existing memories.
 */
export async function resolveContradictions(
  deps: LifecycleDeps,
  contradictions: string[],
  ownerId: string,
  ownerNamespace: string,
): Promise<{ resolved: number }> {
  if (!contradictions.length) return { resolved: 0 };

  let resolved = 0;
  const existing = await deps.storage.queryMemoriesByFilter({
    owner_id: ownerId,
    owner_namespace: ownerNamespace,
    status: "active",
  });

  for (const contradiction of contradictions) {
    // Extract the old fact from format "NEW: X (supersedes OLD: Y)"
    const oldMatch = contradiction.match(/(?:supersedes?\s*(?:OLD)?:?\s*)(.+?)(?:\)|$)/i);
    if (!oldMatch?.[1]) continue;

    const oldFact = oldMatch[1].trim();
    const oldEmbedding = await deps.embedder.embed(oldFact);

    // Find the most similar existing memory to the old fact
    let bestMatch: { memory: MemoryRecord; similarity: number } | null = null;
    for (const mem of existing) {
      if (mem.status !== "active") continue;
      const sim = cosineSimilarity(mem.embedding, oldEmbedding);
      if (sim > 0.75 && (!bestMatch || sim > bestMatch.similarity)) {
        bestMatch = { memory: mem, similarity: sim };
      }
    }

    if (bestMatch) {
      await deps.storage.updateMemory(bestMatch.memory.memory_id, {
        status: "superseded",
        updated_at: Date.now(),
      });
      await deps.storage.insertEvent({
        event_id: randomUUID(),
        memory_id: bestMatch.memory.memory_id,
        event_type: "contradiction",
        event_time: Date.now(),
        details_json: JSON.stringify({
          contradiction_text: contradiction,
          old_content: bestMatch.memory.content.slice(0, 500),
          similarity: bestMatch.similarity,
        }),
      });
      resolved++;
      console.log(`[memory-lancedb-brain] contradiction resolved: superseded "${bestMatch.memory.content.slice(0, 80)}" (sim=${bestMatch.similarity.toFixed(3)})`);
    }
  }

  return { resolved };
}

/**
 * Item 3: Confidence Decay
 * Reduce confidence for memories not used/recalled in a long time.
 * Does NOT archive — just lowers their retrieval score.
 */
export async function applyConfidenceDecay(
  deps: LifecycleDeps,
  config: { halfLifeDays?: number; minConfidence?: number } = {},
  now = Date.now(),
): Promise<{ decayed: number }> {
  const halfLifeDays = config.halfLifeDays ?? 60;
  const minConfidence = config.minConfidence ?? 0.2;
  const halfLifeMs = halfLifeDays * 86_400_000;

  const active = await deps.storage.queryMemoriesByFilter({ status: "active" });
  let decayed = 0;

  for (const memory of active) {
    const daysSinceUsed = (now - memory.last_used_at) / 86_400_000;
    if (daysSinceUsed < 14) continue; // Don't decay recent memories

    // Exponential decay: confidence *= 0.5^(days/halfLife)
    const decayFactor = Math.pow(0.5, (now - memory.last_used_at) / halfLifeMs);
    const newConfidence = Math.max(minConfidence, memory.confidence * decayFactor);

    // Only update if meaningful change (>0.02)
    if (memory.confidence - newConfidence > 0.02) {
      await deps.storage.updateMemory(memory.memory_id, {
        confidence: newConfidence,
        updated_at: now,
      });
      decayed++;
    }
  }

  return { decayed };
}

/**
 * Item 1: Memory Consolidation
 * Uses LLM to merge related memory fragments into clean, cohesive statements.
 */
export async function consolidateMemories(
  deps: LifecycleDeps,
  llmCall: LLMCaller,
  ownerId: string,
  ownerNamespace: string,
): Promise<{ consolidated: number; groupsMerged: number }> {
  const active = await deps.storage.queryMemoriesByFilter({
    owner_id: ownerId,
    owner_namespace: ownerNamespace,
    status: "active",
  });

  if (active.length < 5) return { consolidated: 0, groupsMerged: 0 };

  // Cluster memories by cosine similarity (>0.75 = same cluster)
  const clusters: MemoryRecord[][] = [];
  const assigned = new Set<string>();

  for (const mem of active) {
    if (assigned.has(mem.memory_id)) continue;
    const cluster: MemoryRecord[] = [mem];
    assigned.add(mem.memory_id);

    for (const other of active) {
      if (assigned.has(other.memory_id)) continue;
      if (mem.memory_type !== other.memory_type && mem.memory_type !== "summary" && other.memory_type !== "summary") continue;
      const sim = cosineSimilarity(mem.embedding, other.embedding);
      if (sim > 0.75) {
        cluster.push(other);
        assigned.add(other.memory_id);
      }
    }

    if (cluster.length >= 3) {
      clusters.push(cluster);
    }
  }

  if (clusters.length === 0) return { consolidated: 0, groupsMerged: 0 };

  let consolidated = 0;
  let groupsMerged = 0;

  for (const cluster of clusters) {
    const memoryTexts = cluster.map((m) => `- [${m.memory_type}] ${m.content}`).join("\n");

    try {
      const result = await llmCall(
        `You are a memory consolidation engine. Given related memory fragments about a user, merge them into 1-2 clean, standalone statements that preserve ALL factual information. Return ONLY the merged statements, one per line. No JSON, no markdown, no commentary. Preserve the user's language (Chinese/English).`,
        `Consolidate these related memory fragments into fewer, cleaner statements:\n\n${memoryTexts}`,
      );

      const mergedStatements = result.split("\n").map((l) => l.replace(/^[-•*]\s*/, "").trim()).filter((l) => l.length > 5);
      if (mergedStatements.length === 0) continue;

      // Archive old fragments
      for (const mem of cluster) {
        await deps.storage.updateMemory(mem.memory_id, {
          status: "archived",
          updated_at: Date.now(),
        });
        await deps.storage.insertEvent({
          event_id: randomUUID(),
          memory_id: mem.memory_id,
          event_type: "consolidate",
          event_time: Date.now(),
          details_json: JSON.stringify({ reason: "consolidated_into_new", cluster_size: cluster.length }),
        });
      }

      // Insert consolidated memories
      const maxImportance = Math.max(...cluster.map((m) => m.importance));
      const maxConfidence = Math.max(...cluster.map((m) => m.confidence));

      for (const text of mergedStatements) {
        const embedding = await deps.embedder.embed(text);
        const newMem: MemoryRecord = {
          memory_id: randomUUID(),
          owner_namespace: ownerNamespace,
          owner_id: ownerId,
          agent_id: cluster[0].agent_id,
          memory_scope: cluster[0].memory_scope,
          memory_type: cluster[0].memory_type,
          title: generateTitle(text),
          content: text,
          summary: generateSummary(text),
          tags: JSON.stringify(["consolidated"]),
          importance: Math.min(5, maxImportance + 1),
          confidence: maxConfidence,
          status: "active",
          supersedes_id: "",
          created_at: Date.now(),
          updated_at: Date.now(),
          last_used_at: Date.now(),
          source_session_id: cluster[0].source_session_id,
          embedding,
        };
        await deps.storage.insertMemory(newMem);
        await deps.storage.insertEvent({
          event_id: randomUUID(),
          memory_id: newMem.memory_id,
          event_type: "consolidate",
          event_time: Date.now(),
          details_json: JSON.stringify({ reason: "consolidation_result", source_count: cluster.length }),
        });
      }

      consolidated += cluster.length;
      groupsMerged++;
      console.log(`[memory-lancedb-brain] consolidated ${cluster.length} fragments into ${mergedStatements.length} clean memories`);
    } catch (err) {
      console.warn(`[memory-lancedb-brain] consolidation failed for cluster: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  return { consolidated, groupsMerged };
}

/**
 * Item 4: User Profile Synthesis
 * Aggregates all owner_shared memories into a structured user profile.
 */
export async function synthesizeUserProfile(
  deps: LifecycleDeps,
  llmCall: LLMCaller,
  ownerId: string,
  ownerNamespace: string,
): Promise<{ profileMemoryId: string; profile: string } | null> {
  const active = await deps.storage.queryMemoriesByFilter({
    owner_id: ownerId,
    owner_namespace: ownerNamespace,
    memory_scope: "owner_shared",
    status: "active",
  });

  // Exclude existing profile summaries from input
  const nonProfile = active.filter((m) => !m.content.startsWith("## ") && m.memory_type !== "summary");
  if (nonProfile.length < 3) return null;

  const memoryTexts = nonProfile
    .map((m) => `- [${m.memory_type}] ${m.content}`)
    .join("\n");

  try {
    const profile = await llmCall(
      `You are a user profile synthesizer. Given scattered memory fragments about a user, create a structured profile using markdown headers. Include ONLY what the memories state — do not infer. Preserve the user's language. Use this structure:

## 基本資料 / Identity
## 專長與技能 / Skills & Expertise
## 硬體與環境 / Hardware & Environment
## 偏好與習慣 / Preferences & Habits
## 進行中的專案 / Active Projects
## 溝通風格 / Communication Style

Keep it concise — one line per fact. Omit empty sections.`,
      `Synthesize this user profile from the following memories:\n\n${memoryTexts}`,
    );

    if (!profile || profile.trim().length < 20) return null;

    // Archive any existing profile memory
    const existingProfiles = active.filter((m) =>
      m.memory_type === "summary" && m.content.startsWith("## "),
    );
    for (const old of existingProfiles) {
      await deps.storage.updateMemory(old.memory_id, {
        status: "archived",
        updated_at: Date.now(),
      });
      await deps.storage.insertEvent({
        event_id: randomUUID(),
        memory_id: old.memory_id,
        event_type: "profile_sync",
        event_time: Date.now(),
        details_json: JSON.stringify({ reason: "replaced_by_new_profile" }),
      });
    }

    // Insert new profile as high-importance summary
    const embedding = await deps.embedder.embed(profile.slice(0, 500));
    const profileMem: MemoryRecord = {
      memory_id: randomUUID(),
      owner_namespace: ownerNamespace,
      owner_id: ownerId,
      agent_id: "system",
      memory_scope: "owner_shared",
      memory_type: "summary",
      title: `用戶畫像 / User Profile — ${ownerId}`,
      content: profile,
      summary: `Structured profile for ${ownerId}`,
      tags: JSON.stringify(["profile", "synthesized"]),
      importance: 5,
      confidence: 0.95,
      status: "active",
      supersedes_id: "",
      created_at: Date.now(),
      updated_at: Date.now(),
      last_used_at: Date.now(),
      source_session_id: "profile-synthesis",
      embedding,
    };

    await deps.storage.insertMemory(profileMem);
    await deps.storage.insertEvent({
      event_id: randomUUID(),
      memory_id: profileMem.memory_id,
      event_type: "profile_sync",
      event_time: Date.now(),
      details_json: JSON.stringify({ source_memory_count: nonProfile.length }),
    });

    console.log(`[memory-lancedb-brain] user profile synthesized: ${profile.length} chars from ${nonProfile.length} memories`);
    return { profileMemoryId: profileMem.memory_id, profile };
  } catch (err) {
    console.warn(`[memory-lancedb-brain] profile synthesis failed: ${err instanceof Error ? err.message : String(err)}`);
    return null;
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
  let superseded = 0;
  let supersedeCandidates = 0;
  let firstMemoryId: string | undefined;
  const insertedTypes: string[] = [];

  for (const candidate of candidates) {
    const result = await mergeOrInsertMemory(deps, candidate, mergeConfig);
    if (!firstMemoryId) firstMemoryId = result.memoryId;
    insertedTypes.push(candidate.memory_type);

    if (result.action === "inserted") inserted += 1;
    else merged += 1;

    const supersedeCands = await findSupersedeCandidates(deps, candidate, supersedeConfig);
    supersedeCandidates += supersedeCands.length;
    if (supersedeCands.length > 0) {
      await recordSupersedeEvents(deps, supersedeCands);
      superseded += supersedeCands.length;
    }
  }

  const archive = archiveConfig ? await archiveColdMemories(deps, archiveConfig) : { archivedCount: 0, events: [] };

  return {
    inserted,
    merged,
    superseded,
    supersedeCandidates,
    archived: archive.archivedCount,
    firstMemoryId,
    insertedTypes,
  };
}
