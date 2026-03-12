import assert from "node:assert";
import { randomUUID } from "node:crypto";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { test } from "node:test";

import registerPlugin from "../index.ts";
import { createMemoryBrainContextEngine, createMemoryDistillCommand } from "../src/context-engine.ts";
import { MemoryStorage } from "../src/storage.ts";

const BASE_DB_PATH = "/tmp/memory-lancedb-brain-phase3";

function createFakeEmbedder() {
  return {
    async embed(text) {
      const lower = String(text).toLowerCase();
      const vector = Array.from({ length: 2560 }, () => 0);
      if (lower.includes("docker")) vector[0] = 1;
      if (lower.includes("preference") || lower.includes("偏好")) vector[1] = 1;
      if (lower.includes("summary")) vector[2] = 1;
      return vector;
    },
  };
}

async function setupStorage() {
  const dbPath = `${BASE_DB_PATH}-${randomUUID()}`;
  await rm(dbPath, { recursive: true, force: true });
  await mkdir(dbPath, { recursive: true });
  const storage = await MemoryStorage.connect(dbPath);
  return { dbPath, storage };
}

function createDeps(storage, overrides = {}) {
  return {
    storage,
    embedder: createFakeEmbedder(),
    distiller: {
      async distillTranscript() {
        return {
          session_summary: "Session summary for testing",
          confirmed_facts: ["Docker runs on GB10"],
          decisions: ["Use memory-lancedb-brain"],
          pitfalls: ["Do not expose owner_shared to tiffany-customer"],
          preference_updates: ["Peter prefers direct answers"],
          environment_truths: ["OpenClaw is 2026.3.8"],
          open_loops: ["Implement merge lifecycle"],
          scope_recommendation: "both",
        };
      },
    },
    owners: [{ owner_id: "peter", owner_namespace: "personal", channels: { telegram: "6647202606" } }],
    agentWhitelist: ["main", "plaw-coding-team", "tiffany-ops"],
    retrieval: { mode: "hybrid", minScore: 0.1, hardMinScore: 0.05 },
    sessionStates: new Map(),
    sessionKeyIndex: new Map(),
    lastSessionByChannel: new Map(),
    ...overrides,
  };
}

async function insertMemory(storage, partial) {
  const now = Date.now();
  await storage.insertMemory({
    memory_id: randomUUID(),
    owner_namespace: "personal",
    owner_id: "peter",
    agent_id: "main",
    memory_scope: "owner_shared",
    memory_type: "fact",
    title: "Default title",
    content: "Default content",
    summary: "Default summary",
    tags: "[]",
    importance: 3,
    confidence: 0.8,
    status: "active",
    supersedes_id: "",
    created_at: now,
    updated_at: now,
    last_used_at: now,
    source_session_id: "seed-session",
    embedding: Array.from({ length: 2560 }, () => 0.1),
    ...partial,
  });
}

test("Phase 3: registerContextEngine + registerCommand from plugin entry", async () => {
  const { dbPath } = await setupStorage();
  const registrations = { contextEngine: null, command: null, tools: 0 };
  try {
    const api = {
      pluginConfig: {
        dbPath,
        owners: [{ owner_id: "peter", owner_namespace: "personal", channels: { telegram: "6647202606" } }],
      },
      resolvePath(input) { return input; },
      registerTool() { registrations.tools += 1; },
      registerContextEngine(id, factory) { registrations.contextEngine = { id, factory }; },
      registerCommand(command) { registrations.command = command; },
      on() {},
      logger: { info() {}, error(message) { throw new Error(message); } },
    };

    registerPlugin(api);
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.strictEqual(registrations.contextEngine.id, "memory-lancedb-brain");
    assert.strictEqual(registrations.command.name, "memory");
    assert.strictEqual(registrations.tools, 6);
  } finally {
    await rm(dbPath, { recursive: true, force: true });
  }
});

test("Phase 3: assemble injects owner_shared and agent_local memories", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    await insertMemory(storage, {
      title: "Docker layout",
      content: "Docker runs on GB10 and Windows uses Docker Desktop",
      summary: "Docker layout summary",
      memory_scope: "owner_shared",
      embedding: await createFakeEmbedder().embed("Docker runs on GB10 and Windows uses Docker Desktop"),
    });
    await insertMemory(storage, {
      title: "Agent note",
      content: "This is a local agent implementation detail",
      summary: "Local detail",
      memory_scope: "agent_local",
      agent_id: "main",
      embedding: await createFakeEmbedder().embed("local agent implementation detail"),
    });

    const deps = createDeps(storage);
    deps.sessionStates.set("s1", {
      sessionId: "s1",
      sessionKey: "key-s1",
      agentId: "main",
      owner: { ownerId: "peter", ownerNamespace: "personal" },
      channelId: "telegram",
      sessionFile: undefined,
      staging: ["Docker on GB10"],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const result = await engine.assemble({
      sessionId: "s1",
      messages: [{ role: "user", content: "How is Docker arranged?" }],
    });

    assert.match(result.systemPromptAddition, /OWNER SHARED MEMORY/);
    assert.match(result.systemPromptAddition, /AGENT LOCAL MEMORY/);
    assert.match(result.systemPromptAddition, /Docker/);
  } finally {
    await rm(dbPath, { recursive: true, force: true });
  }
});

test("Phase 3: ingest and afterTurn stage valuable snippets", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    const deps = createDeps(storage);
    const engine = createMemoryBrainContextEngine(deps);

    await engine.ingest({
      sessionId: "s2",
      message: { role: "user", content: "This is an important implementation detail about memory boundaries and Docker deployment." },
    });

    await engine.afterTurn({
      sessionId: "s2",
      sessionFile: "/tmp/nonexistent-session.jsonl",
      messages: [
        { role: "user", content: "short" },
        { role: "assistant", content: "This is another sufficiently long summary that should be staged for future distillation." },
      ],
      prePromptMessageCount: 0,
      autoCompactionSummary: "This turn discussed memory boundaries and owner-shared rules in detail.",
    });

    const state = deps.sessionStates.get("s2");
    assert.ok(state.staging.length >= 2);
  } finally {
    await rm(dbPath, { recursive: true, force: true });
  }
});

test("Phase 3: compact writes distilled memories and /memory distill triggers best-effort flow", async () => {
  const { dbPath, storage } = await setupStorage();
  const sessionFile = `/tmp/memory-lancedb-brain-session-${randomUUID()}.jsonl`;
  try {
    await writeFile(
      sessionFile,
      [
        JSON.stringify({ role: "user", content: "Docker runs on GB10." }),
        JSON.stringify({ role: "assistant", content: "We decided to use memory-lancedb-brain." }),
      ].join("\n"),
      "utf8",
    );

    const deps = createDeps(storage);
    deps.sessionStates.set("s3", {
      sessionId: "s3",
      sessionKey: "key-s3",
      agentId: "main",
      owner: { ownerId: "peter", ownerNamespace: "personal" },
      channelId: "telegram",
      sessionFile,
      staging: ["Peter prefers direct answers"],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });
    deps.lastSessionByChannel.set("telegram", "s3");

    const engine = createMemoryBrainContextEngine(deps);
    const compactResult = await engine.compact({ sessionId: "s3", sessionFile, force: true, currentTokenCount: 200 });
    assert.strictEqual(compactResult.ok, true);
    assert.strictEqual(compactResult.compacted, true);

    const rowsAfterCompact = await storage.queryMemoriesByFilter({ owner_id: "peter" });
    assert.ok(rowsAfterCompact.length >= 5);

    const command = createMemoryDistillCommand(deps, engine);
    const commandResult = await command.handler({ args: "distill", channel: "telegram" });
    assert.match(commandResult.text, /inserted/);
  } finally {
    await rm(dbPath, { recursive: true, force: true });
    await rm(sessionFile, { force: true });
  }
});

test("Phase 3: prepareSubagentSpawn and onSubagentEnded promote child staging to parent", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    const deps = createDeps(storage);
    deps.sessionStates.set("parent-session", {
      sessionId: "parent-session",
      sessionKey: "parent-key",
      agentId: "main",
      owner: { ownerId: "peter", ownerNamespace: "personal" },
      channelId: "telegram",
      sessionFile: undefined,
      staging: ["parent memory"],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });
    deps.sessionKeyIndex.set("parent-key", "parent-session");

    const engine = createMemoryBrainContextEngine(deps);
    await engine.prepareSubagentSpawn({ parentSessionKey: "parent-key", childSessionKey: "child-key" });

    const childId = deps.sessionKeyIndex.get("child-key") ?? "child-key";
    const childState = deps.sessionStates.get(childId);
    childState.staging.push("child discovered important implementation detail");
    deps.sessionStates.set(childId, childState);

    await engine.onSubagentEnded({ childSessionKey: "child-key", reason: "completed" });
    const parent = deps.sessionStates.get("parent-session");
    assert.ok(parent.staging.some((line) => line.includes("subagent:completed")));
  } finally {
    await rm(dbPath, { recursive: true, force: true });
  }
});
