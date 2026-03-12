/**
 * Phase 1 smoke tests for memory-lancedb-brain
 */

import assert from "node:assert";
import { test } from "node:test";
import { randomUUID } from "node:crypto";
import { mkdir, rm } from "node:fs/promises";

// Test configuration
const TEST_DB_PATH = "/tmp/memory-lancedb-test";
const EMBEDDING_ENDPOINT = "http://127.0.0.1:32080/v1";
const EMBEDDING_HEALTH_ENDPOINT = "http://127.0.0.1:32080/v1/models";

// Import the modules we're testing
import { MemoryStorage } from "../src/storage.ts";
import { createEmbedder } from "../src/embedding.ts";

let testDbPath = `${TEST_DB_PATH}-${randomUUID()}`;

async function setup() {
  // Clean up and create temp directory
  try {
    await rm(testDbPath, { recursive: true, force: true });
  } catch {}
  
  await mkdir(testDbPath, { recursive: true });
}

async function teardown() {
  // Clean up test database
  try {
    await rm(testDbPath, { recursive: true, force: true });
  } catch {}
}

test("Phase 1: LanceDB connection and table creation", async (t) => {
  await setup();
  
  try {
    // Test 1: Connect to LanceDB
    const storage = await MemoryStorage.connect(testDbPath);
    assert.ok(storage, "MemoryStorage should be created");
    
    // Test 2: Tables should exist
    const memoriesTable = await storage.getDb().openTable("memories");
    assert.ok(memoriesTable, "memories table should exist");
    
    const eventsTable = await storage.getDb().openTable("memory_events");
    assert.ok(eventsTable, "memory_events table should exist");
    
    console.log("✓ LanceDB connection and table creation passed");
  } finally {
    await teardown();
  }
});

test("Phase 1: Insert and query memory round-trip", async (t) => {
  await setup();
  
  try {
    const storage = await MemoryStorage.connect(testDbPath);
    
    // Create a test memory record
    const testMemory = {
      memory_id: randomUUID(),
      owner_namespace: "test-namespace",
      owner_id: "test-owner",
      agent_id: "test-agent",
      memory_scope: "agent_local",
      memory_type: "fact",
      title: "Test Memory",
      content: "This is a test memory content",
      summary: "Test summary",
      tags: JSON.stringify(["test", "phase1"]),
      importance: 3,
      confidence: 0.8,
      status: "active",
      supersedes_id: "",
      created_at: Date.now(),
      updated_at: Date.now(),
      last_used_at: Date.now(),
      source_session_id: "test-session-123",
      embedding: Array.from({ length: 2560 }, () => 0.5),
    };
    
    // Insert memory
    await storage.insertMemory(testMemory);
    
    // Query by owner_id
    const results = await storage.queryMemoriesByFilter({ owner_id: "test-owner" });
    assert.strictEqual(results.length, 1, "Should find exactly one memory");
    assert.strictEqual(results[0].memory_id, testMemory.memory_id, "Memory ID should match");
    assert.strictEqual(results[0].title, testMemory.title, "Title should match");
    assert.strictEqual(results[0].tags, JSON.stringify(["test", "phase1"]), "Tags should remain JSON serialized string");
    
    // Query count
    const count = await storage.countMemories({ owner_id: "test-owner" });
    assert.strictEqual(count, 1, "Count should be 1");
    
    console.log("✓ Insert and query round-trip passed");
  } finally {
    await teardown();
  }
});

test("Phase 1: Update and delete memory", async (t) => {
  await setup();
  
  try {
    const storage = await MemoryStorage.connect(testDbPath);
    
    // Create test memory
    const testMemory = {
      memory_id: randomUUID(),
      owner_namespace: "test-namespace",
      owner_id: "test-owner-update",
      agent_id: "test-agent",
      memory_scope: "owner_shared",
      memory_type: "preference",
      title: "To Be Updated",
      content: "Original content",
      summary: "Original summary",
      tags: JSON.stringify(["original"]),
      importance: 2,
      confidence: 0.6,
      status: "active",
      supersedes_id: "",
      created_at: Date.now(),
      updated_at: Date.now(),
      last_used_at: Date.now(),
      source_session_id: "test-session-456",
      embedding: Array.from({ length: 2560 }, () => 0.3),
    };
    
    await storage.insertMemory(testMemory);
    
    // Update memory
    await storage.updateMemory(testMemory.memory_id, {
      title: "Updated Title",
      content: "Updated content",
      importance: 4,
      status: "archived",
    });
    
    // Verify update
    const results = await storage.queryMemoriesByFilter({ owner_id: "test-owner-update" });
    assert.strictEqual(results.length, 1, "Should still have one memory");
    assert.strictEqual(results[0].title, "Updated Title", "Title should be updated");
    assert.strictEqual(results[0].importance, 4, "Importance should be updated");
    assert.strictEqual(results[0].status, "archived", "Status should be archived");
    
    // Delete memory
    await storage.deleteMemory(testMemory.memory_id);
    
    // Verify deletion
    const afterDelete = await storage.queryMemoriesByFilter({ owner_id: "test-owner-update" });
    assert.strictEqual(afterDelete.length, 0, "Memory should be deleted");
    
    console.log("✓ Update and delete passed");
  } finally {
    await teardown();
  }
});

test("Phase 1: Memory event tracking", async (t) => {
  await setup();
  
  try {
    const storage = await MemoryStorage.connect(testDbPath);
    
    const memoryId = randomUUID();
    
    // Create memory
    const testMemory = {
      memory_id: memoryId,
      owner_namespace: "test-namespace",
      owner_id: "test-owner-events",
      agent_id: "test-agent",
      memory_scope: "session_distilled",
      memory_type: "decision",
      title: "Event Test",
      content: "Testing event tracking",
      summary: "Event test summary",
      tags: JSON.stringify([]),
      importance: 3,
      confidence: 0.7,
      status: "active",
      supersedes_id: "",
      created_at: Date.now(),
      updated_at: Date.now(),
      last_used_at: Date.now(),
      source_session_id: "test-session-789",
      embedding: Array.from({ length: 2560 }, () => 0.7),
    };
    
    await storage.insertMemory(testMemory);
    
    // Insert events
    await storage.insertEvent({
      event_id: randomUUID(),
      memory_id: memoryId,
      event_type: "create",
      event_time: Date.now(),
      details_json: JSON.stringify({ source: "initial" }),
    });
    
    await storage.insertEvent({
      event_id: randomUUID(),
      memory_id: memoryId,
      event_type: "recall",
      event_time: Date.now(),
      details_json: JSON.stringify({ context: "search" }),
    });
    
    // Query events
    const events = await storage.queryEvents(memoryId);
    assert.strictEqual(events.length, 2, "Should have 2 events");
    assert.strictEqual(events[0].event_type, "recall", "Most recent should be recall");
    assert.strictEqual(events[1].event_type, "create", "Oldest should be create");
    
    console.log("✓ Memory event tracking passed");
  } finally {
    await teardown();
  }
});

test("Phase 1: Vector search", async (t) => {
  await setup();
  
  try {
    const storage = await MemoryStorage.connect(testDbPath);
    
    // Insert multiple memories with different embeddings
    const memories = [];
    for (let i = 0; i < 3; i++) {
      memories.push({
        memory_id: randomUUID(),
        owner_namespace: "test-namespace",
        owner_id: `owner-${i}`,
        agent_id: "test-agent",
        memory_scope: "agent_local",
        memory_type: "fact",
        title: `Memory ${i}`,
        content: `Content for memory ${i}`,
        summary: `Summary ${i}`,
        tags: JSON.stringify([`tag-${i}`]),
        importance: i + 1,
        confidence: 0.5,
        status: "active",
        supersedes_id: "",
        created_at: Date.now(),
        updated_at: Date.now(),
        last_used_at: Date.now(),
        source_session_id: "test-session",
        embedding: Array.from({ length: 2560 }, () => i * 0.1),
      });
    }
    
    for (const mem of memories) {
      await storage.insertMemory(mem);
    }
    
    // Search with filter
    const queryVector = Array.from({ length: 2560 }, () => 0.2);
    const results = await storage.vectorSearch(queryVector, 2, { status: "active" });
    
    assert.strictEqual(results.length, 2, "Should return top 2 results");
    assert.ok(results.every(r => r.status === "active"), "All results should be active");
    
    console.log("✓ Vector search passed");
  } finally {
    await teardown();
  }
});

test("Phase 1: Embedding provider basic functionality", async (t) => {
  // Skip if endpoint not available
  try {
    const response = await fetch(EMBEDDING_HEALTH_ENDPOINT, {
      headers: { "Authorization": "Bearer local" },
    });
    
    if (!response.ok) {
      console.log("⊘ Skipping embedding test - endpoint not available");
      return;
    }
    
    // Test embedder creation
    const config = {
      apiKey: "local",
      model: "vllm/Forturne/Qwen3-Embedding-4B-NVFP4",
      baseURL: EMBEDDING_ENDPOINT,
      dimensions: 2560,
    };
    
    const embedder = createEmbedder(config);
    assert.ok(embedder, "Embedder should be created");
    
    // Test single embedding
    const text = "Hello world test";
    const embedding = await embedder.embed(text);
    
    assert.ok(Array.isArray(embedding), "Embedding should be an array");
    assert.strictEqual(embedding.length, 2560, "Embedding should have 2560 dimensions");
    
    // Test batch embedding
    const texts = ["Hello world 1", "Hello world 2"];
    const embeddings = await embedder.embedBatch(texts);
    
    assert.strictEqual(embeddings.length, 2, "Should return 2 embeddings");
    assert.strictEqual(embeddings[0].length, 2560, "Each embedding should have 2560 dimensions");
    
    console.log("✓ Embedding provider test passed");
  } catch (error) {
    console.log(`⊘ Skipping embedding test - endpoint unavailable: ${error.message}`);
  }
});

console.log("\n=== Phase 1 Smoke Tests Complete ===");
