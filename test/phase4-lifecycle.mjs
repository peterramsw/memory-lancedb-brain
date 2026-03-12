/**
 * Phase 4 Lifecycle Tests
 * Tests for merge, supersede detection, and archive operations
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert";
import { InMemoryEmbedder, InMemoryStorage } from "./fixtures.js";
import { buildCandidateMemories, processLifecycle, findSupersedeCandidates, archiveColdMemories, recordSupersedeEvents } from "../src/lifecycle.js";

describe("Phase 4: Lifecycle", () => {
  let storage;
  let embedder;

  beforeEach(() => {
    storage = new InMemoryStorage();
    embedder = new InMemoryEmbedder(10);
  });

  describe("merge", () => {
    it("should merge highly similar memories (similarity > 0.92)", async () => {
      const now = Date.now();
      
      // Insert first memory with a simple embedding
      await storage.insertMemory({
        memory_id: "mem-1",
        owner_namespace: "test",
        owner_id: "owner-1",
        agent_id: "agent-1",
        memory_scope: "owner_shared",
        memory_type: "fact",
        title: "Test Fact",
        content: "This is the original fact content",
        summary: "Original summary",
        tags: JSON.stringify([]),
        importance: 3,
        confidence: 0.8,
        status: "active",
        supersedes_id: "",
        created_at: now,
        updated_at: now,
        last_used_at: now,
        source_session_id: "session-1",
        embedding: [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.6, 0.7, 0.8],
      });

      // Create a highly similar candidate
      const candidates = await buildCandidateMemories(
        { storage, embedder },
        { sessionId: "session-2", ownerId: "owner-1", ownerNamespace: "test", agentId: "agent-1" },
        {
          session_summary: "",
          confirmed_facts: ["This is the original fact content with minor additions"],
          decisions: [],
          pitfalls: [],
          preference_updates: [],
          environment_truths: [],
          open_loops: [],
          scope_recommendation: "owner_shared",
        },
        now + 1000,
      );

      if (candidates.length === 0) {
        throw new Error("buildCandidateMemories returned no candidates");
      }

      // Process through lifecycle
      const result = await processLifecycle(
        { storage, embedder },
        candidates,
        { threshold: 0.92 },
        { similarityThreshold: 0.80 },
      );

      // Should have merged, not inserted new
      assert.strictEqual(result.inserted, 0, "Should not insert new memory");
      assert.strictEqual(result.merged, 1, "Should merge 1 memory");

      // Verify only one active memory exists
      const allMemories = await storage.queryMemoriesByFilter({ status: "active" });
      assert.strictEqual(allMemories.length, 1, "Should have only 1 active memory");

      // Verify merge event was recorded
      const events = await storage.queryEvents({ memory_id: "mem-1" });
      const mergeEvent = events.find((e) => e.event_type === "merge");
      assert.ok(mergeEvent, "Should have merge event");
    });
  });

  describe("supersede candidate", () => {
    it("should identify supersede candidates without changing status", async () => {
      const now = Date.now();

      // Insert existing memory with a specific embedding
      await storage.insertMemory({
        memory_id: "mem-old",
        owner_namespace: "test",
        owner_id: "owner-1",
        agent_id: "agent-1",
        memory_scope: "owner_shared",
        memory_type: "decision",
        title: "Old Decision",
        content: "We decided to use approach A",
        summary: "Decision for approach A",
        tags: JSON.stringify([]),
        importance: 3,
        confidence: 0.8,
        status: "active",
        supersedes_id: "",
        created_at: now - 10000,
        updated_at: now - 10000,
        last_used_at: now - 10000,
        source_session_id: "session-old",
        embedding: [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
      });

      // Create a similar but different candidate with slightly different embedding
      // This should be above 0.80 similarity threshold but below 0.92 merge threshold
      const candidates = await buildCandidateMemories(
        { storage, embedder },
        { sessionId: "session-new", ownerId: "owner-1", ownerNamespace: "test", agentId: "agent-1" },
        {
          session_summary: "",
          confirmed_facts: [],
          decisions: ["Actually, we should use approach B instead"],
          pitfalls: [],
          preference_updates: [],
          environment_truths: [],
          open_loops: [],
          scope_recommendation: "owner_shared",
        },
        now + 1000, // Make sure candidate has newer timestamp
      );

      if (candidates.length === 0) {
        throw new Error("buildCandidateMemories returned no candidates");
      }

      // Set embedding to be similar but NOT identical - similarity should be > 0.80 but < 0.92
      // Using [0.7, 0.2, 0.7, 0.2, 0.7, 0.2, 0.7, 0.2, 0.7, 0.2] vs [0.8, 0.8, ...]
      // This gives approximately 0.874 similarity
      candidates[0].embedding = [0.7, 0.2, 0.7, 0.2, 0.7, 0.2, 0.7, 0.2, 0.7, 0.2];

      // Find supersede candidates
      const supersedeList = await findSupersedeCandidates(
        { storage, embedder },
        candidates[0],
        { similarityThreshold: 0.80 },
      );

      // Debug logging
      console.log('DEBUG: supersedeList length:', supersedeList.length);
      if (supersedeList.length > 0) {
        console.log('DEBUG: Found supersede candidate:', supersedeList[0].existingMemory.memory_id);
        console.log('DEBUG: Similarity:', supersedeList[0].similarity);
      }

      // Should identify the old decision as a supersede candidate
      assert.strictEqual(supersedeList.length, 1, "Should find 1 supersede candidate");
      assert.strictEqual(supersedeList[0].existingMemory.memory_id, "mem-old");

      // Old memory should still be active
      const oldMemory = await storage.getMemory("mem-old");
      assert.strictEqual(oldMemory.status, "active", "Old memory should remain active");

      // Record supersede events (this is what processLifecycle does)
      await recordSupersedeEvents(
        { storage, embedder },
        supersedeList,
      );

      // Supersede event should be recorded
      const events = await storage.queryEvents({ memory_id: "mem-old" });
      const supersedeEvent = events.find((e) => e.event_type === "supersede");
      assert.ok(supersedeEvent, "Should have supersede event");
    });
  });

  describe("archive", () => {
    it("should archive cold memories (importance <= 1, last_used_at > 90 days)", async () => {
      const now = Date.now();
      const ninetyOneDaysAgo = now - 91 * 24 * 60 * 60 * 1000;

      // Insert a cold memory
      await storage.insertMemory({
        memory_id: "mem-cold",
        owner_namespace: "test",
        owner_id: "owner-1",
        agent_id: "agent-1",
        memory_scope: "owner_shared",
        memory_type: "fact",
        title: "Cold Fact",
        content: "This fact is no longer relevant",
        summary: "Cold fact summary",
        tags: JSON.stringify([]),
        importance: 1,
        confidence: 0.5,
        status: "active",
        supersedes_id: "",
        created_at: ninetyOneDaysAgo,
        updated_at: ninetyOneDaysAgo,
        last_used_at: ninetyOneDaysAgo,
        source_session_id: "session-old",
        embedding: [0.3, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5, 0.6],
      });

      // Run archive sweep
      const archiveResult = await archiveColdMemories({ storage, embedder }, { maxImportance: 1, daysInactive: 90 });

      // Should archive the cold memory
      assert.strictEqual(archiveResult.archivedCount, 1, "Should archive 1 memory");

      // Verify memory status changed to archived
      const archivedMemory = await storage.getMemory("mem-cold");
      assert.strictEqual(archivedMemory.status, "archived", "Memory should be archived");

      // Archive event should be recorded
      const events = await storage.queryEvents({ memory_id: "mem-cold" });
      const archiveEvent = events.find((e) => e.event_type === "archive");
      assert.ok(archiveEvent, "Should have archive event");
    });
  });

  describe("compact integration", () => {
    it("should run full lifecycle flow during compact", async () => {
      const now = Date.now();

      // First session creates some memories
      const session1 = await buildCandidateMemories(
        { storage, embedder },
        { sessionId: "session-1", ownerId: "owner-1", ownerNamespace: "test", agentId: "agent-1" },
        {
          session_summary: "Session 1 summary",
          confirmed_facts: ["User prefers TypeScript over JavaScript"],
          decisions: ["Use React for UI"],
          pitfalls: [],
          preference_updates: [],
          environment_truths: [],
          open_loops: [],
          scope_recommendation: "owner_shared",
        },
        now,
      );

      await processLifecycle(
        { storage, embedder },
        session1,
        { threshold: 0.92 },
        { similarityThreshold: 0.80 },
      );

      // Second session has similar content
      const session2 = await buildCandidateMemories(
        { storage, embedder },
        { sessionId: "session-2", ownerId: "owner-1", ownerNamespace: "test", agentId: "agent-1" },
        {
          session_summary: "Session 2 summary (similar to 1)",
          confirmed_facts: ["User prefers TypeScript over JavaScript (reconfirmed)"],
          decisions: ["Use React for UI (confirmed again)"],
          pitfalls: [],
          preference_updates: [],
          environment_truths: [],
          open_loops: [],
          scope_recommendation: "owner_shared",
        },
        now + 1000,
      );

      // Process through lifecycle
      const result = await processLifecycle(
        { storage, embedder },
        session2,
        { threshold: 0.92 },
        { similarityThreshold: 0.80 },
      );

      // Should merge rather than duplicate
      assert.ok(result.merged >= 1, "Should merge at least 1 memory");

      // Events should be recorded
      const allEvents = await storage.queryEvents({});
      const mergeEvents = allEvents.filter((e) => e.event_type === "merge");
      assert.ok(mergeEvents.length > 0, "Should have merge events");
    });
  });
});
