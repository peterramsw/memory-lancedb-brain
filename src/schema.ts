/**
 * LanceDB schema definitions for memory-lancedb-brain
 */

// Memory record types
export type MemoryScope = "owner_shared" | "agent_local" | "session_distilled";
export type MemoryType = "fact" | "preference" | "decision" | "pitfall" | "status" | "todo" | "summary" | "correction" | "best_practice" | "goal" | "episode";
export type MemoryStatus = "active" | "archived" | "superseded";
export type MemoryEventType = "create" | "merge" | "promote" | "archive" | "supersede" | "recall" | "distill" | "consolidate" | "contradiction" | "profile_sync";

/**
 * MemoryRecord represents a single memory entry in LanceDB
 */
export interface MemoryRecord {
  memory_id: string; // UUID
  owner_namespace: string;
  owner_id: string;
  agent_id: string;
  memory_scope: MemoryScope;
  memory_type: MemoryType;
  title: string;
  content: string;
  summary: string;
  tags: string; // JSON array serialized
  importance: number; // 1-5
  confidence: number; // 0-1
  status: MemoryStatus;
  supersedes_id: string; // empty string for null
  created_at: number; // epoch ms
  updated_at: number; // epoch ms
  last_used_at: number; // epoch ms
  source_session_id: string;
  source: string; // how this memory was created: "distill" | "manual" | "consolidate" | "synthesize" | "ingest:*"
  embedding: number[]; // dim 2560
}

/**
 * MemoryEventRecord represents lifecycle events for memories
 */
export interface MemoryEventRecord {
  event_id: string; // UUID
  memory_id: string;
  event_type: MemoryEventType;
  event_time: number; // epoch ms
  details_json: string; // JSON object serialized
}
