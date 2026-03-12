/**
 * LanceDB storage layer for memory operations
 */

import type { Connection, Table } from "@lancedb/lancedb";
import * as lancedb from "@lancedb/lancedb";
import type { MemoryEventRecord, MemoryRecord, MemoryScope, MemoryStatus, MemoryType } from "./schema.js";

export interface MemoryFilters {
  memory_id?: string;
  owner_id?: string;
  owner_namespace?: string;
  agent_id?: string;
  memory_scope?: MemoryScope;
  memory_type?: MemoryType;
  status?: MemoryStatus;
}

export interface MemoryTables {
  memories: Table;
  memoryEvents: Table;
}

function escapeSqlLiteral(value: string): string {
  return value.replace(/'/g, "''");
}

function buildWhereClause(filters?: MemoryFilters): string | null {
  if (!filters) return null;

  const conditions: string[] = [];
  if (filters.memory_id) conditions.push(`memory_id = '${escapeSqlLiteral(filters.memory_id)}'`);
  if (filters.owner_id) conditions.push(`owner_id = '${escapeSqlLiteral(filters.owner_id)}'`);
  if (filters.owner_namespace) conditions.push(`owner_namespace = '${escapeSqlLiteral(filters.owner_namespace)}'`);
  if (filters.agent_id) conditions.push(`agent_id = '${escapeSqlLiteral(filters.agent_id)}'`);
  if (filters.memory_scope) {
    conditions.push(`memory_scope = '${escapeSqlLiteral(filters.memory_scope)}'`);
  }
  if (filters.memory_type) {
    conditions.push(`memory_type = '${escapeSqlLiteral(filters.memory_type)}'`);
  }
  if (filters.status) conditions.push(`status = '${escapeSqlLiteral(filters.status)}'`);

  return conditions.length > 0 ? conditions.join(" AND ") : null;
}

function normalizeMemoryRecord(record: any): MemoryRecord {
  return {
    ...record,
    tags: typeof record.tags === "string" ? record.tags : JSON.stringify(record.tags ?? []),
    embedding: Array.isArray(record.embedding)
      ? record.embedding.map(Number)
      : Array.from(record.embedding ?? [], Number),
  } as MemoryRecord;
}

function normalizeEventRecord(record: any): MemoryEventRecord {
  return {
    ...record,
    details_json:
      typeof record.details_json === "string"
        ? record.details_json
        : JSON.stringify(record.details_json ?? {}),
  } as MemoryEventRecord;
}

function toTableRow(record: MemoryRecord): Record<string, unknown> {
  return {
    ...record,
    embedding: Array.from(record.embedding, Number),
  };
}

function toEventRow(record: MemoryEventRecord): Record<string, unknown> {
  return {
    ...record,
  };
}

function seedMemoryRecord(): MemoryRecord {
  const now = Date.now();
  return {
    memory_id: "__seed__",
    owner_namespace: "seed",
    owner_id: "seed",
    agent_id: "seed",
    memory_scope: "agent_local",
    memory_type: "fact",
    title: "seed",
    content: "seed",
    summary: "seed",
    tags: "[]",
    importance: 1,
    confidence: 0,
    status: "active",
    supersedes_id: "",
    created_at: now,
    updated_at: now,
    last_used_at: now,
    source_session_id: "seed",
    embedding: Array.from({ length: 2560 }, () => 0),
  };
}

function seedEventRecord(): MemoryEventRecord {
  return {
    event_id: "__seed_event__",
    memory_id: "__seed__",
    event_type: "create",
    event_time: Date.now(),
    details_json: "{}",
  };
}

export async function connectDb(dbPath: string): Promise<Connection> {
  return lancedb.connect(dbPath);
}

export async function ensureTables(db: Connection): Promise<MemoryTables> {
  let memories: Table;
  let memoryEvents: Table;

  try {
    memories = await db.openTable("memories");
  } catch {
    memories = await db.createTable("memories", [toTableRow(seedMemoryRecord())]);
    await memories.delete("memory_id = '__seed__'");
  }

  try {
    memoryEvents = await db.openTable("memory_events");
  } catch {
    memoryEvents = await db.createTable("memory_events", [toEventRow(seedEventRecord())]);
    await memoryEvents.delete("event_id = '__seed_event__'");
  }

  return { memories, memoryEvents };
}

export async function insertMemory(table: Table, record: MemoryRecord): Promise<void> {
  await table.add([toTableRow(record)]);
}

export async function queryMemoriesByFilter(
  table: Table,
  filters?: MemoryFilters,
): Promise<MemoryRecord[]> {
  let query = table.query();
  const where = buildWhereClause(filters);
  if (where) query = query.where(where);
  const rows = await query.toArray();
  return rows.map(normalizeMemoryRecord);
}

export async function updateMemory(
  table: Table,
  memory_id: string,
  updates: Partial<MemoryRecord>,
): Promise<void> {
  const values: Record<string, string | number | boolean> = {
    updated_at: Date.now(),
  };

  for (const [key, value] of Object.entries(updates)) {
    if (typeof value === "undefined") continue;
    if (key === "embedding") continue;
    if (key === "tags" && typeof value !== "string") {
      values[key] = JSON.stringify(value);
      continue;
    }
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      values[key] = value;
    }
  }

  await table.update({
    where: `memory_id = '${escapeSqlLiteral(memory_id)}'`,
    values,
  });
}

export async function deleteMemory(table: Table, memory_id: string): Promise<void> {
  await table.delete(`memory_id = '${escapeSqlLiteral(memory_id)}'`);
}

export async function insertEvent(eventsTable: Table, event: MemoryEventRecord): Promise<void> {
  await eventsTable.add([toEventRow(event)]);
}

export async function queryEvents(eventsTable: Table, memory_id: string): Promise<MemoryEventRecord[]> {
  const rows = await eventsTable
    .query()
    .where(`memory_id = '${escapeSqlLiteral(memory_id)}'`)
    .toArray();

  return rows
    .map(normalizeEventRecord)
    .sort((a, b) => b.event_time - a.event_time);
}

export async function countMemories(table: Table, filters?: MemoryFilters): Promise<number> {
  const where = buildWhereClause(filters);
  return where ? table.countRows(where) : table.countRows();
}

export class MemoryStorage {
  private constructor(
    private readonly db: Connection,
    private readonly tables: MemoryTables,
  ) {}

  static async connect(dbPath: string): Promise<MemoryStorage> {
    const db = await connectDb(dbPath);
    const tables = await ensureTables(db);
    return new MemoryStorage(db, tables);
  }

  getDb(): Connection {
    return this.db;
  }

  get memoriesTable(): Table {
    return this.tables.memories;
  }

  get eventsTable(): Table {
    return this.tables.memoryEvents;
  }

  async insertMemory(record: MemoryRecord): Promise<void> {
    return insertMemory(this.tables.memories, record);
  }

  async queryMemoriesByFilter(filters?: MemoryFilters): Promise<MemoryRecord[]> {
    return queryMemoriesByFilter(this.tables.memories, filters);
  }

  async updateMemory(memoryId: string, updates: Partial<MemoryRecord>): Promise<void> {
    return updateMemory(this.tables.memories, memoryId, updates);
  }

  async deleteMemory(memoryId: string): Promise<void> {
    return deleteMemory(this.tables.memories, memoryId);
  }

  async insertEvent(event: MemoryEventRecord): Promise<void> {
    return insertEvent(this.tables.memoryEvents, event);
  }

  async queryEvents(memoryId: string): Promise<MemoryEventRecord[]> {
    return queryEvents(this.tables.memoryEvents, memoryId);
  }

  async countMemories(filters?: MemoryFilters): Promise<number> {
    return countMemories(this.tables.memories, filters);
  }

  async vectorSearch(queryVector: number[] | Float32Array, limit = 10, filters?: MemoryFilters): Promise<MemoryRecord[]> {
    let query = this.tables.memories.search(Array.from(queryVector, Number)).limit(limit);
    const where = buildWhereClause(filters);
    if (where) query = query.where(where);
    const rows = await query.toArray();
    return rows.map(normalizeMemoryRecord);
  }
}
