import assert from "node:assert";
import { randomUUID } from "node:crypto";
import { mkdir, rm } from "node:fs/promises";
import { test } from "node:test";

import { MemoryStorage } from "../src/storage.ts";
import { registerAllMemoryTools } from "../src/tools.ts";
import { canAccessScope, resolveOwnerFromContext } from "../src/owners.ts";

const BASE_DB_PATH = "/tmp/memory-lancedb-brain-phase2";

async function safeRm(path) {
  try { await rm(path, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 }); } catch {}
}

function createFakeEmbedder() {
  return {
    async embed(text) {
      const lower = String(text).toLowerCase();
      const base = Array.from({ length: 2560 }, () => 0);
      if (lower.includes("docker")) base[0] = 1;
      if (lower.includes("prefer") || lower.includes("偏好")) base[1] = 1;
      if (lower.includes("todo") || lower.includes("待辦")) base[2] = 1;
      if (lower.includes("memory")) base[3] = 1;
      return base;
    },
  };
}

function createRuntime(agentId = "main") {
  return {
    agentId,
    sessionId: `session-${randomUUID()}`,
    sessionKey: `session-${randomUUID()}`,
    messageChannel: "telegram",
    requesterSenderId: "0000000000",
    senderIsOwner: true,
  };
}

function makeMockApi() {
  const registrations = [];
  return {
    registrations,
    registerTool(tool, opts) {
      registrations.push({ tool, opts });
    },
  };
}

function materializeTools(registrations, runtimeCtx) {
  const map = new Map();
  for (const entry of registrations) {
    const built = typeof entry.tool === "function" ? entry.tool(runtimeCtx) : entry.tool;
    map.set(built.name, built);
  }
  return map;
}

async function setupStorage() {
  const dbPath = `${BASE_DB_PATH}-${randomUUID()}`;
  await safeRm(dbPath);
  await mkdir(dbPath, { recursive: true });
  const storage = await MemoryStorage.connect(dbPath);
  return { dbPath, storage };
}

test("Phase 2: tool registration count", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    const api = makeMockApi();
    registerAllMemoryTools(api, {
      storage,
      embedder: createFakeEmbedder(),
      owners: [
        {
          owner_id: "test-user",
          owner_namespace: "personal",
          channels: { telegram: "0000000000" },
        },
      ],
      agentWhitelist: ["main", "plaw-coding-team", "tiffany-ops"],
      retrieval: { mode: "hybrid" },
    });
    assert.strictEqual(api.registrations.length, 6);
  } finally {
    await safeRm(dbPath);
  }
});

test("Phase 2: memory_store -> memory_recall round-trip", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    const api = makeMockApi();
    registerAllMemoryTools(api, {
      storage,
      embedder: createFakeEmbedder(),
      owners: [{ owner_id: "test-user", owner_namespace: "personal", channels: { telegram: "0000000000" } }],
      agentWhitelist: ["main", "plaw-coding-team", "tiffany-ops"],
      retrieval: { mode: "hybrid", minScore: 0.1, hardMinScore: 0.05 },
    });

    const tools = materializeTools(api.registrations, createRuntime("main"));
    const storeResult = await tools.get("memory_store").execute("1", {
      text: "Docker runs on GB10 and Windows uses Docker Desktop",
      importance: 4,
      scope: "owner_shared",
      category: "fact",
    });
    assert.ok(storeResult.details.memory_id);

    const recallResult = await tools.get("memory_recall").execute("2", {
      query: "Docker GB10",
      limit: 5,
      scope: "owner_shared",
    });
    assert.ok(recallResult.details.count >= 1);
    assert.match(recallResult.details.memories[0].content, /Docker/);
  } finally {
    await safeRm(dbPath);
  }
});

test("Phase 2: memory_update + list + stats + forget", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    const api = makeMockApi();
    registerAllMemoryTools(api, {
      storage,
      embedder: createFakeEmbedder(),
      owners: [{ owner_id: "test-user", owner_namespace: "personal", channels: { telegram: "0000000000" } }],
      agentWhitelist: ["main", "plaw-coding-team", "tiffany-ops"],
      retrieval: { mode: "hybrid", minScore: 0.1, hardMinScore: 0.05 },
    });
    const tools = materializeTools(api.registrations, createRuntime("main"));

    const first = await tools.get("memory_store").execute("1", {
      text: "I prefer concise technical answers",
      scope: "owner_shared",
      category: "preference",
    });
    const memoryId = first.details.memory_id;

    const updated = await tools.get("memory_update").execute("2", {
      memoryId,
      text: "I prefer concise but complete technical answers",
      importance: 5,
      category: "preference",
    });
    assert.match(updated.details.memory.content, /complete technical answers/);
    assert.strictEqual(updated.details.memory.importance, 5);

    const listResult = await tools.get("memory_list").execute("3", {
      scope: "owner_shared",
      limit: 10,
    });
    assert.ok(listResult.details.count >= 1);

    const statsResult = await tools.get("memory_stats").execute("4", {});
    assert.ok(statsResult.details.total >= 1);
    assert.ok(statsResult.details.byScope.owner_shared >= 1);

    const forgetResult = await tools.get("memory_forget").execute("5", { memoryId });
    assert.strictEqual(forgetResult.details.memory_id, memoryId);

    const afterDelete = await tools.get("memory_list").execute("6", { scope: "owner_shared" });
    assert.strictEqual(afterDelete.details.count, 0);
  } finally {
    await safeRm(dbPath);
  }
});

test("Phase 2: tiffany-customer hard deny for owner_shared", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    const api = makeMockApi();
    registerAllMemoryTools(api, {
      storage,
      embedder: createFakeEmbedder(),
      owners: [{ owner_id: "test-user", owner_namespace: "personal", channels: { telegram: "0000000000" } }],
      agentWhitelist: ["main", "plaw-coding-team", "tiffany-ops"],
      retrieval: { mode: "hybrid" },
    });
    const tools = materializeTools(api.registrations, createRuntime("tiffany-customer"));

    const denied = await tools.get("memory_store").execute("1", {
      text: "Should not store to owner_shared",
      scope: "owner_shared",
    });
    assert.match(denied.content[0].text, /Access denied/);
  } finally {
    await safeRm(dbPath);
  }
});

test("Phase 2: recall excludes archived and superseded by default", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    const api = makeMockApi();
    registerAllMemoryTools(api, {
      storage,
      embedder: createFakeEmbedder(),
      owners: [{ owner_id: "test-user", owner_namespace: "personal", channels: { telegram: "0000000000" } }],
      agentWhitelist: ["main", "plaw-coding-team", "tiffany-ops"],
      retrieval: { mode: "hybrid", minScore: 0.1, hardMinScore: 0.05 },
    });
    const tools = materializeTools(api.registrations, createRuntime("main"));

    const active = await tools.get("memory_store").execute("1", {
      text: "Docker environment truth active",
      scope: "owner_shared",
      category: "fact",
    });
    const archived = await tools.get("memory_store").execute("2", {
      text: "Docker archived memory",
      scope: "owner_shared",
      category: "fact",
    });
    const superseded = await tools.get("memory_store").execute("3", {
      text: "Docker superseded memory",
      scope: "owner_shared",
      category: "fact",
    });

    await storage.updateMemory(archived.details.memory_id, { status: "archived" });
    await storage.updateMemory(superseded.details.memory_id, { status: "superseded" });

    const recall = await tools.get("memory_recall").execute("4", {
      query: "Docker",
      scope: "owner_shared",
      limit: 10,
    });

    const ids = recall.details.memories.map((memory) => memory.memory_id);
    assert.ok(ids.includes(active.details.memory_id));
    assert.ok(!ids.includes(archived.details.memory_id));
    assert.ok(!ids.includes(superseded.details.memory_id));
  } finally {
    await safeRm(dbPath);
  }
});

test("Phase 2: owner mapping helper + whitelist helper", async () => {
  const owner = resolveOwnerFromContext(
    { senderId: "0000000000", messageChannel: "telegram", agentId: "main", senderIsOwner: true },
    [{ owner_id: "test-user", owner_namespace: "personal", channels: { telegram: "0000000000" } }],
  );
  assert.deepStrictEqual(owner, { ownerId: "test-user", ownerNamespace: "personal" });
  assert.strictEqual(canAccessScope("main", "owner_shared", ["main"]), true);
  assert.strictEqual(canAccessScope("tiffany-customer", "owner_shared", ["tiffany-customer"]), false);
});
