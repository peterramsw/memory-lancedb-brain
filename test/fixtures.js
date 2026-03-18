/**
 * Test fixtures for memory-lancedb-brain
 */

import { randomUUID } from "node:crypto";

/**
 * In-memory storage implementation for testing lifecycle operations
 */
export class InMemoryStorage {
  constructor() {
    this.memories = new Map();
    this.events = new Map();
  }

  async connect() {
    // No-op for in-memory storage
  }

  async disconnect() {
    // No-op
  }

  async ensureTables() {
    // No-op
  }

  async getTable(name) {
    if (name === "memories") return { type: "memories", data: this.memories };
    if (name === "memoryEvents") return { type: "memoryEvents", data: this.events };
    throw new Error(`Table ${name} not found`);
  }

  async insertMemory(memory) {
    this.memories.set(memory.memory_id, { ...memory });
  }

  async getMemory(memoryId) {
    return this.memories.get(memoryId) || null;
  }

  async updateMemory(memoryId, updates) {
    const existing = this.memories.get(memoryId);
    if (!existing) throw new Error(`Memory ${memoryId} not found`);
    Object.assign(existing, updates);
  }

  async deleteMemory(memoryId) {
    this.memories.delete(memoryId);
  }

  async queryMemoriesByFilter(filters = {}) {
    const results = [];
    for (const memory of this.memories.values()) {
      let matches = true;
      if (filters.memory_id && memory.memory_id !== filters.memory_id) matches = false;
      if (filters.owner_id && memory.owner_id !== filters.owner_id) matches = false;
      if (filters.owner_namespace && memory.owner_namespace !== filters.owner_namespace) matches = false;
      if (filters.agent_id && memory.agent_id !== filters.agent_id) matches = false;
      if (filters.memory_scope && memory.memory_scope !== filters.memory_scope) matches = false;
      if (filters.memory_type && memory.memory_type !== filters.memory_type) matches = false;
      if (filters.status && memory.status !== filters.status) matches = false;
      if (matches) results.push(memory);
    }
    return results;
  }

  async insertEvent(event) {
    this.events.set(event.event_id, { ...event });
  }

  async queryEvents(filters = {}) {
    const results = [];
    for (const event of this.events.values()) {
      let matches = true;
      if (filters.memory_id && event.memory_id !== filters.memory_id) matches = false;
      if (filters.event_type && event.event_type !== filters.event_type) matches = false;
      if (matches) results.push(event);
    }
    return results;
  }

  async getEvent(eventId) {
    return this.events.get(eventId) || null;
  }

  async clear() {
    this.memories.clear();
    this.events.clear();
  }
}

/**
 * Simple in-memory embedder for testing
 * Generates deterministic embeddings based on text hash
 */
export class InMemoryEmbedder {
  constructor(dimensions = 10) {
    this.dimensions = dimensions;
  }

  async embed(text) {
    // Generate deterministic pseudo-random embedding based on text
    const hash = this.hashString(text);
    const embedding = [];
    for (let i = 0; i < this.dimensions; i++) {
      embedding.push((hash + i) % 1000 / 1000);
    }
    return embedding;
  }

  async embedBatch(texts) {
    return Promise.all(texts.map((t) => this.embed(t)));
  }

  hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return Math.abs(hash);
  }
}

/**
 * Similarity calculation for testing
 */
export function cosineSimilarity(a, b) {
  if (a.length !== b.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Create similar embeddings for testing merge scenarios
 */
export function createSimilarEmbeddings(baseText, similarity = 0.95, dimensions = 10) {
  const baseEmbedder = new InMemoryEmbedder(dimensions);
  const baseEmbedding = baseEmbedder.embedSync?.(baseText) || [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.6, 0.7, 0.8];
  
  const result = [];
  for (let i = 0; i < dimensions; i++) {
    result.push(baseEmbedding[i] * similarity + (Math.random() * (1 - similarity)));
  }
  return result;
}
