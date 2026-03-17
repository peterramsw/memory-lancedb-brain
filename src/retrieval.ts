/**
 * Hybrid retrieval with vector search + simplified keyword scoring + optional reranking
 */

import type { MemoryRecord, MemoryScope } from "./schema.js";
import type { MemoryFilters, MemoryStorage } from "./storage.js";

export interface RetrievalResult {
  memory: MemoryRecord;
  score: number;
  reasons: string[];
}

export interface HybridRetrieveParams {
  query: string;
  ownerId?: string;
  ownerNamespace?: string;
  agentId?: string;
  scope?: MemoryScope;
  category?: MemoryRecord["memory_type"];
  limit?: number;
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
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9\u4e00-\u9fff]+/i)
    .filter(Boolean);
}

function buildFilters(params: HybridRetrieveParams): MemoryFilters {
  const filters: MemoryFilters = {
    status: "active",
  };
  if (params.ownerId) filters.owner_id = params.ownerId;
  if (params.ownerNamespace) filters.owner_namespace = params.ownerNamespace;
  if (params.scope) filters.memory_scope = params.scope;
  if (params.category) filters.memory_type = params.category;
  if (params.scope === "agent_local" || params.scope === "session_distilled") {
    filters.agent_id = params.agentId;
  }
  return filters;
}

function keywordScore(query: string, memory: MemoryRecord): { score: number; matchedTerms: string[] } {
  const queryTerms = tokenize(query);
  if (queryTerms.length === 0) return { score: 0, matchedTerms: [] };

  const haystack = tokenize(`${memory.title} ${memory.summary} ${memory.content}`);
  const counts = new Map<string, number>();
  for (const token of haystack) counts.set(token, (counts.get(token) ?? 0) + 1);

  let score = 0;
  const matchedTerms: string[] = [];
  for (const term of queryTerms) {
    const matches = counts.get(term) ?? 0;
    if (matches > 0) {
      matchedTerms.push(term);
      score += matches;
    }
  }

  return {
    score: queryTerms.length > 0 ? score / queryTerms.length : 0,
    matchedTerms,
  };
}

async function keywordSearch(
  storage: MemoryStorage,
  query: string,
  filters: MemoryFilters,
  limit: number,
): Promise<Array<{ memory: MemoryRecord; score: number; matchedTerms: string[] }>> {
  const rows = await storage.queryMemoriesByFilter(filters);
  return rows
    .map((memory) => {
      const { score, matchedTerms } = keywordScore(query, memory);
      return { memory, score, matchedTerms };
    })
    .filter((row) => row.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

async function rerankResults(
  query: string,
  results: RetrievalResult[],
  rerankEndpoint?: string,
  rerankApiKey?: string,
  rerankModel?: string,
): Promise<RetrievalResult[]> {
  if (!rerankEndpoint || results.length === 0) return results;

  try {
    const response = await fetch(rerankEndpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(rerankApiKey ? { Authorization: `Bearer ${rerankApiKey}` } : {}),
      },
      body: JSON.stringify({
        model: rerankModel ?? "vllm/Forturne/Qwen3-Reranker-4B-NVFP4",
        query,
        documents: results.map((result) => ({
          id: result.memory.memory_id,
          text: `${result.memory.title}\n${result.memory.summary}\n${result.memory.content}`,
        })),
        top_n: results.length,
      }),
    });

    if (!response.ok) return results;
    const payload = await response.json();
    const ranked = payload?.results ?? payload?.data ?? [];
    if (!Array.isArray(ranked) || ranked.length === 0) return results;

    const rerankScoreById = new Map<string, number>();
    for (const item of ranked) {
      const id = item.document?.id ?? item.id;
      const score = Number(item.relevance_score ?? item.score ?? 0);
      if (typeof id === "string") {
        rerankScoreById.set(id, score);
      } else if (typeof item.index === "number" && item.index < results.length) {
        rerankScoreById.set(results[item.index].memory.memory_id, score);
      }
    }

    // Blend: 60% rerank score + 40% original score (not just re-sort)
    return [...results]
      .map(r => {
        const rerankScore = rerankScoreById.get(r.memory.memory_id);
        return {
          ...r,
          score: rerankScore != null
            ? rerankScore * 0.6 + r.score * 0.4
            : r.score * 0.8,  // penalize unreturned candidates slightly
        };
      })
      .sort((a, b) => b.score - a.score);
  } catch {
    return results;
  }
}

export async function hybridRetrieve(
  storage: MemoryStorage,
  embedder: { embed(text: string): Promise<number[]> },
  params: HybridRetrieveParams,
): Promise<RetrievalResult[]> {
  const {
    query,
    limit = 5,
    mode = "hybrid",
    vectorWeight = 0.7,
    bm25Weight = 0.3,
    minScore = 0.25,
    hardMinScore = 0.1,
    candidatePoolSize = Math.max(10, limit * 5),
  } = params;

  const filters = buildFilters(params);
  const queryEmbedding = await embedder.embed(query);

  const vectorCandidates =
    mode === "keyword"
      ? []
      : (await storage.vectorSearch(queryEmbedding, candidatePoolSize, filters)).map((memory, index) => ({
          memory,
          score: Math.max(0.2, 1 - index * 0.08),
        }));

  const keywordCandidates =
    mode === "vector"
      ? []
      : await keywordSearch(storage, query, filters, candidatePoolSize);

  const merged = new Map<string, RetrievalResult>();

  for (const row of vectorCandidates) {
    merged.set(row.memory.memory_id, {
      memory: row.memory,
      score: row.score * vectorWeight,
      reasons: ["vector"],
    });
  }

  for (const row of keywordCandidates) {
    const existing = merged.get(row.memory.memory_id);
    const keywordComponent = Math.min(1, row.score) * bm25Weight;
    if (existing) {
      existing.score += keywordComponent;
      existing.reasons.push("keyword");
    } else {
      merged.set(row.memory.memory_id, {
        memory: row.memory,
        score: keywordComponent,
        reasons: ["keyword"],
      });
    }
  }

  let results = [...merged.values()]
    .filter((row) => row.score >= hardMinScore)
    .sort((a, b) => b.score - a.score)
    .filter((row) => row.score >= minScore)
    .slice(0, candidatePoolSize);

  if (params.rerank) {
    results = await rerankResults(
      query,
      results,
      params.rerankEndpoint,
      params.rerankApiKey,
      params.rerankModel,
    );
  }

  return results.slice(0, limit);
}

export function detectCategory(text: string): MemoryRecord["memory_type"] {
  const value = text.toLowerCase();
  if (/(喜歡|偏好|prefer|preference|love|hate|習慣)/i.test(value)) return "preference";
  if (/(決定|改用|decide|選擇|決策)/i.test(value)) return "decision";
  if (/(避免|不要|坑|pitfall|問題|bug)/i.test(value)) return "pitfall";
  if (/(不對|不是|wrong|correct|糾正|應該是)/i.test(value)) return "correction";
  if (/(有效|成功|這樣做|best practice|這招|管用)/i.test(value)) return "best_practice";
  if (/(待辦|todo|要做|需要處理|follow up)/i.test(value)) return "todo";
  if (/(目標|goal|project|專案|計畫|ongoing)/i.test(value)) return "goal";
  if (/(摘要|總結|summary)/i.test(value)) return "summary";
  if (/(狀態|status|進度)/i.test(value)) return "status";
  return "fact";
}

export function mapCategory(input?: string): MemoryRecord["memory_type"] | undefined {
  if (!input) return undefined;
  const value = input.trim().toLowerCase();
  const allowed: MemoryRecord["memory_type"][] = [
    "fact",
    "preference",
    "decision",
    "pitfall",
    "status",
    "todo",
    "summary",
    "goal",
    "correction",
    "best_practice",
  ];
  if ((allowed as string[]).includes(value)) return value as MemoryRecord["memory_type"];
  return detectCategory(value);
}

export function generateTitle(text: string): string {
  const cleaned = text.trim().replace(/\s+/g, " ");
  const firstLine = cleaned.split(/\n+/)[0] ?? cleaned;
  return firstLine.length <= 80 ? firstLine : `${firstLine.slice(0, 77)}...`;
}

export function generateSummary(text: string): string {
  const cleaned = text.trim().replace(/\s+/g, " ");
  return cleaned.length <= 200 ? cleaned : `${cleaned.slice(0, 197)}...`;
}
