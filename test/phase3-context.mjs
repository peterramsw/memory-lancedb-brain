import assert from "node:assert";
import { randomUUID } from "node:crypto";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { test } from "node:test";

import registerPlugin from "../index.ts";
import { createMemoryBrainContextEngine, createMemoryDistillCommand } from "../src/context-engine.ts";
import { createDistiller } from "../src/distill.ts";
import { MemoryStorage } from "../src/storage.ts";

const BASE_DB_PATH = "/tmp/memory-lancedb-brain-phase3";

// LanceDB may hold file handles briefly after close; swallow cleanup errors
// so they don't mask real test failures (especially on slow CI disks).
async function safeRm(path) {
  try { await rm(path, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 }); } catch {}
}

function createFakeEmbedder() {
  return {
    async embed(text) {
      const lower = String(text).toLowerCase();
      const vector = Array.from({ length: 2560 }, () => 0);
      if (lower.includes("docker")) vector[0] = 1;
      if (lower.includes("preference") || lower.includes("偏好")) vector[1] = 1;
      if (lower.includes("summary")) vector[2] = 1;
      if (lower.includes("share-echo-41")) vector[3] = 1;
      if (lower.includes("local-only-77")) vector[4] = 1;
      return vector;
    },
  };
}

async function setupStorage() {
  const dbPath = `${BASE_DB_PATH}-${randomUUID()}`;
  await safeRm(dbPath);
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
          preference_updates: ["User prefers direct answers"],
          environment_truths: ["OpenClaw is 2026.3.8"],
          open_loops: ["Implement merge lifecycle"],
          corrections: [],
          best_practices: [],
          style_observations: [],
          expertise_signals: [],
          active_goals: [],
          contradictions: [],
          scope_recommendation: "both",
        };
      },
    },
    owners: [{ owner_id: "test-user", owner_namespace: "personal", channels: { telegram: "0000000000" } }],
    agentWhitelist: ["main", "plaw-coding-team", "tiffany-ops"],
    retrieval: { mode: "hybrid", minScore: 0.1, hardMinScore: 0.05 },
    sessionStates: new Map(),
    sessionKeyIndex: new Map(),
    lastSessionByChannel: new Map(),
    diagnostics: {
      pluginId: "memory-lancedb-brain",
      dbPath: BASE_DB_PATH,
      initializedAt: Date.now(),
      recentWarnings: [],
      recentErrors: [],
    },
    ...overrides,
  };
}

async function insertMemory(storage, partial) {
  const now = Date.now();
  await storage.insertMemory({
    memory_id: randomUUID(),
    owner_namespace: "personal",
    owner_id: "test-user",
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
        owners: [{ owner_id: "test-user", owner_namespace: "personal", channels: { telegram: "0000000000" } }],
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
    assert.strictEqual(registrations.tools, 7);
  } finally {
    await safeRm(dbPath);
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
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      channelId: "telegram",
      sessionFile: undefined,
      staging: ["Docker deployment on GB10 infrastructure?"],
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
    await safeRm(dbPath);
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
    await safeRm(dbPath);
  }
});

test("Phase 3: compact writes distilled memories and /memory distill triggers best-effort flow", async () => {
  const { dbPath, storage } = await setupStorage();
  const sessionFile = `/tmp/memory-lancedb-brain-session-${randomUUID()}.jsonl`;
  try {
    await writeFile(
      sessionFile,
      [
        JSON.stringify({ type: "message", message: { role: "user", content: "Docker runs on GB10. The deployment uses memory-lancedb-brain as the primary memory plugin." } }),
        JSON.stringify({ type: "message", message: { role: "assistant", content: "We decided to use memory-lancedb-brain for long-term memory storage and retrieval." } }),
      ].join("\n"),
      "utf8",
    );

    const deps = createDeps(storage);
    deps.sessionStates.set("s3", {
      sessionId: "s3",
      sessionKey: "key-s3",
      agentId: "main",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      channelId: "telegram",
      sessionFile,
      staging: ["User prefers direct answers"],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });
    deps.lastSessionByChannel.set("telegram", "s3");

    const engine = createMemoryBrainContextEngine(deps);
    const compactResult = await engine.compact({ sessionId: "s3", sessionFile, force: true, currentTokenCount: 200 });
    assert.strictEqual(compactResult.ok, true);
    // compacted=true means messages were truncated; with only 2 messages the
    // file may not exceed the 40% token budget, so we check insertedCount instead.
    assert.ok(compactResult.result?.insertedCount > 0, "compact should have inserted memories");

    const rowsAfterCompact = await storage.queryMemoriesByFilter({ owner_id: "test-user" });
    assert.ok(rowsAfterCompact.length >= 5);

    const command = createMemoryDistillCommand(deps, engine);
    const commandResult = await command.handler({ args: "distill", channel: "telegram" });
    assert.match(commandResult.text, /inserted/);
  } finally {
    await safeRm(dbPath);
    await safeRm(sessionFile);
  }
});

test("Phase 3 regression: compact sanitizes transcript and respects distill token budget", async () => {
  const { dbPath, storage } = await setupStorage();
  const sessionFile = `/tmp/memory-lancedb-brain-session-${randomUUID()}.jsonl`;
  let capturedTranscript = "";
  try {
    await writeFile(
      sessionFile,
      [
        JSON.stringify({
          type: "message",
          message: {
            role: "user",
            content: "A new session was started via /new or /reset. Execute your Session Startup sequence now - read the required files before responding to the user.",
          },
        }),
        JSON.stringify({
          type: "message",
          message: {
            role: "assistant",
            content: "你好，我是 peter-365。",
          },
        }),
        JSON.stringify({
          type: "message",
          message: {
            role: "user",
            content: [
              {
                type: "text",
                text: `Conversation info (untrusted metadata):\n\`\`\`json\n{"message_id":"264"}\n\`\`\`\n\n明天王省強要去佰鼎。${" 補充資訊".repeat(200)}`,
              },
            ],
          },
        }),
        JSON.stringify({
          type: "message",
          message: {
            role: "assistant",
            content: "收到，明天王省強要去佰鼎。",
          },
        }),
      ].join("\n"),
      "utf8",
    );

    const deps = createDeps(storage, {
      autoDistill: {
        enabled: true,
        triggers: ["onSessionEnd"],
        minStagingLength: 1,
        tokenBudget: 120,
        onSubagentEnded: false,
        onSessionEnd: true,
        onReset: false,
        onNew: false,
      },
      distiller: {
        async distillTranscript(transcript) {
          capturedTranscript = transcript;
          return {
            session_summary: "明天王省強要去佰鼎",
            confirmed_facts: ["用戶提供的工作安排: 明天王省強要去佰鼎。"],
            decisions: [],
            pitfalls: [],
            preference_updates: [],
            environment_truths: [],
            open_loops: [],
            corrections: [],
            best_practices: [],
            style_observations: [],
            expertise_signals: [],
            active_goals: [],
            contradictions: [],
            scope_recommendation: "owner_shared",
          };
        },
      },
    });
    deps.sessionStates.set("s3-budget", {
      sessionId: "s3-budget",
      sessionKey: "key-s3-budget",
      agentId: "peter-365",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      channelId: "telegram",
      sessionFile,
      staging: ["明天王省強要去佰鼎"],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const compactResult = await engine.compact({ sessionId: "s3-budget", sessionFile, force: true, currentTokenCount: 5000 });
    assert.strictEqual(compactResult.ok, true);
    assert.ok(capturedTranscript.length > 0, "distiller should receive a transcript");
    assert.doesNotMatch(capturedTranscript, /A new session was started via \/new or \/reset/);
    assert.doesNotMatch(capturedTranscript, /Conversation info \(untrusted metadata\)/);
    assert.ok(Math.ceil(capturedTranscript.length / 4) <= 120, `transcript should respect budget, got ~${Math.ceil(capturedTranscript.length / 4)} tokens`);
  } finally {
    await safeRm(dbPath);
    await safeRm(sessionFile);
  }
});

test("Phase 3 regression: compact splits oversized dropped messages into multiple episode records without truncation", async () => {
  const { dbPath, storage } = await setupStorage();
  const sessionFile = `/tmp/memory-lancedb-brain-session-${randomUUID()}.jsonl`;
  const headMarker = "LOSSLESS-HEAD-4411";
  const tailMarker = "LOSSLESS-TAIL-9927";
  const oversized = `${headMarker} ${"tool-output ".repeat(2500)} ${tailMarker}`;

  try {
    await writeFile(
      sessionFile,
      [
        JSON.stringify({ type: "message", message: { role: "user", content: oversized } }),
        JSON.stringify({ type: "message", message: { role: "assistant", content: "Recent reply that should remain in the trimmed session file." } }),
      ].join("\n"),
      "utf8",
    );

    const deps = createDeps(storage);
    deps.sessionStates.set("lossless-split", {
      sessionId: "lossless-split",
      sessionKey: "key-lossless-split",
      agentId: "main",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      channelId: "telegram",
      sessionFile,
      staging: ["Need to preserve the full tool output across compaction."],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const compactResult = await engine.compact({ sessionId: "lossless-split", sessionFile, force: true, currentTokenCount: 2000 });
    assert.strictEqual(compactResult.ok, true);
    assert.strictEqual(compactResult.compacted, true);

    const episodes = (await storage.queryMemoriesByFilter({
      owner_id: "test-user",
      owner_namespace: "personal",
      memory_type: "episode",
      source_session_id: "lossless-split",
      status: "active",
    })).sort((a, b) => (a.created_at || 0) - (b.created_at || 0));

    assert.ok(episodes.length > 1, "oversized dropped message should be split into multiple episode rows");

    const joinedEpisodeContent = episodes.map((memory) => memory.content).join("\n");
    assert.match(joinedEpisodeContent, new RegExp(headMarker));
    assert.match(joinedEpisodeContent, new RegExp(tailMarker));
  } finally {
    await safeRm(dbPath);
    await safeRm(sessionFile);
  }
});

test("Phase 3 regression: assemble recalls stored episode chunks after compaction", async () => {
  const { dbPath, storage } = await setupStorage();
  const sessionFile = `/tmp/memory-lancedb-brain-session-${randomUUID()}.jsonl`;
  const recallMarker = "EPISODE-RECALL-5518";
  const oversized = `${"continuity ".repeat(1800)} ${recallMarker} recall-tail`;

  try {
    await writeFile(
      sessionFile,
      [
        JSON.stringify({ type: "message", message: { role: "user", content: oversized } }),
        JSON.stringify({ type: "message", message: { role: "assistant", content: "Latest assistant turn should remain after compaction." } }),
      ].join("\n"),
      "utf8",
    );

    const deps = createDeps(storage);
    deps.sessionStates.set("episode-recall", {
      sessionId: "episode-recall",
      sessionKey: "key-episode-recall",
      agentId: "main",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      channelId: "telegram",
      sessionFile,
      staging: ["We need to continue the earlier compacted context."],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const compactResult = await engine.compact({ sessionId: "episode-recall", sessionFile, force: true, currentTokenCount: 2000 });
    assert.strictEqual(compactResult.ok, true);
    assert.strictEqual(compactResult.compacted, true);

    const assembleResult = await engine.assemble({
      sessionId: "episode-recall",
      messages: [{ role: "user", content: `Can you continue from ${recallMarker}?` }],
    });

    assert.match(assembleResult.systemPromptAddition, /Relevant_Past_Context/);
    assert.match(assembleResult.systemPromptAddition, new RegExp(recallMarker));
  } finally {
    await safeRm(dbPath);
    await safeRm(sessionFile);
  }
});

test("Phase 3 regression: heuristic fallback preserves operational fact and blocks startup pollution", async () => {
  const { dbPath, storage } = await setupStorage();
  const sessionFile = `/tmp/memory-lancedb-brain-session-${randomUUID()}.jsonl`;
  try {
    await writeFile(
      sessionFile,
      [
        JSON.stringify({
          type: "message",
          message: {
            role: "user",
            content: "A new session was started via /new or /reset. Execute your Session Startup sequence now - read the required files before responding to the user.",
          },
        }),
        JSON.stringify({
          type: "message",
          message: {
            role: "assistant",
            content: "Hey. I just came online. Who am I? Who are you?",
          },
        }),
        JSON.stringify({
          type: "message",
          message: {
            role: "user",
            content: [
              {
                type: "text",
                text: "Conversation info (untrusted metadata):\n```json\n{\"message_id\":\"264\"}\n```\n\n明天王省強要去佰鼎。",
              },
            ],
          },
        }),
        JSON.stringify({
          type: "message",
          message: {
            role: "assistant",
            content: "收到，明天（3/21）王省強要去佰鼎。",
          },
        }),
      ].join("\n"),
      "utf8",
    );

    const deps = createDeps(storage, {
      distiller: createDistiller({
        distillImpl: async () => {
          throw new Error("LLM API error: 400 Bad Request");
        },
      }),
    });
    deps.sessionStates.set("s3-fallback", {
      sessionId: "s3-fallback",
      sessionKey: "key-s3-fallback",
      agentId: "peter-365",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      channelId: "telegram",
      sessionFile,
      staging: ["明天王省強要去佰鼎"],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const compactResult = await engine.compact({ sessionId: "s3-fallback", sessionFile, force: true, currentTokenCount: 400 });
    assert.strictEqual(compactResult.ok, true);
    assert.ok(compactResult.result?.insertedCount > 0, "fallback should still insert useful memories");

    const rows = await storage.queryMemoriesByFilter({ owner_id: "test-user", status: "active" });
    assert.ok(rows.some((memory) => /王省強|佰鼎/.test(memory.content)), "operational fact should remain recallable");
    assert.ok(rows.every((memory) => !/A new session was started via \/new or \/reset/.test(memory.content)), "startup boilerplate must not be persisted");
    assert.ok(rows.every((memory) => !/Conversation info \(untrusted metadata\)/.test(memory.content)), "metadata wrapper must not be persisted");
  } finally {
    await safeRm(dbPath);
    await safeRm(sessionFile);
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
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
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
    await safeRm(dbPath);
  }
});

test("Phase 3 regression: compact fails closed when session has no owner", async () => {
  const { dbPath, storage } = await setupStorage();
  const sessionFile = `/tmp/memory-lancedb-brain-session-${randomUUID()}.jsonl`;
  try {
    await writeFile(
      sessionFile,
      [
        JSON.stringify({ type: "message", message: { role: "user", content: "Docker runs on GB10. The deployment uses memory-lancedb-brain as the primary memory plugin." } }),
        JSON.stringify({ type: "message", message: { role: "assistant", content: "We decided to use memory-lancedb-brain for long-term memory storage and retrieval." } }),
      ].join("\n"),
      "utf8",
    );

    const deps = createDeps(storage, { owners: [] });
    // Session without owner AND no configured owners — simulates missing owner context
    deps.sessionStates.set("no-owner", {
      sessionId: "no-owner",
      sessionKey: "key-no-owner",
      agentId: "main",
      owner: undefined,
      channelId: "telegram",
      sessionFile,
      staging: [],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const result = await engine.compact({ sessionId: "no-owner", sessionFile, force: true, currentTokenCount: 200 });

    // Must fail-closed: no memories written, clear error
    assert.strictEqual(result.ok, false, "compact should fail when owner is missing");
    assert.match(result.reason, /Missing owner context/);

    // Verify no memories were written
    const allMemories = await storage.queryMemoriesByFilter({});
    assert.strictEqual(allMemories.length, 0, "No memories should be written without owner context");
  } finally {
    await safeRm(dbPath);
    await safeRm(sessionFile);
  }
});

test("Phase 3 regression: no hardcoded personal names in src/ defaults", async () => {
  const { readdir: readdirAsync, readFile: readFileAsync } = await import("node:fs/promises");
  const { join: joinPath } = await import("node:path");

  const srcDir = joinPath(import.meta.dirname, "..", "src");
  const files = await readdirAsync(srcDir);
  const tsFiles = files.filter(f => f.endsWith(".ts"));

  const personalNamePattern = /(?:["'`])(?:peter|alice|bob|john|jane)(?:["'`])/i;
  const violations = [];

  for (const file of tsFiles) {
    const content = await readFileAsync(joinPath(srcDir, file), "utf-8");
    const lines = content.split("\n");
    for (let i = 0; i < lines.length; i++) {
      // Skip comments
      const trimmed = lines[i].trim();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;
      if (personalNamePattern.test(lines[i])) {
        violations.push(`${file}:${i + 1}: ${lines[i].trim()}`);
      }
    }
  }

  assert.strictEqual(violations.length, 0, `src/ contains hardcoded personal names as defaults:\n${violations.join("\n")}`);
});

test("Phase 3: assemble uses current user turn and preserves cross-agent owner_shared isolation", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    await insertMemory(storage, {
      title: "Cross-agent shared marker",
      content: "跨 agent 共用測試：共享代號是 share-echo-41",
      summary: "share-echo-41",
      memory_scope: "owner_shared",
      embedding: await createFakeEmbedder().embed("share-echo-41"),
    });
    await insertMemory(storage, {
      title: "Main local marker",
      content: "跨 agent 本地測試：只有 main 應看到 local-only-77",
      summary: "local-only-77",
      memory_scope: "agent_local",
      agent_id: "main",
      embedding: await createFakeEmbedder().embed("local-only-77"),
    });

    const deps = createDeps(storage);
    deps.sessionStates.set("cross-agent", {
      sessionId: "cross-agent",
      sessionKey: "key-cross-agent",
      agentId: "peter-365",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      messageChannel: "telegram",
      requesterSenderId: "0000000000",
      agentAccountId: "bot-account",
      senderIsOwner: true,
      channelId: "telegram",
      sessionFile: undefined,
      staging: ['{"role":"assistant","content":[{"type":"toolCall","id":"call_1"}]}'],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const result = await engine.assemble({
      sessionId: "cross-agent",
      messages: [{ role: "user", content: "還記得 share-echo-41 嗎？那 local-only-77 呢？" }],
    });

    assert.match(result.systemPromptAddition, /share-echo-41/);
    assert.doesNotMatch(result.systemPromptAddition, /local-only-77/);
    assert.strictEqual(deps.diagnostics.lastAssemble?.query.includes("share-echo-41"), true);
  } finally {
    await safeRm(dbPath);
  }
});

test("Phase 3: assemble ignores synthetic /new startup text and falls back to staged user query", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    await insertMemory(storage, {
      title: "Bridge token",
      content: "跨 agent live smoke token is bridge-8841",
      summary: "bridge-8841",
      memory_scope: "owner_shared",
      embedding: await createFakeEmbedder().embed("bridge-8841"),
    });

    const deps = createDeps(storage);
    deps.sessionStates.set("startup-filter", {
      sessionId: "startup-filter",
      sessionKey: "key-startup-filter",
      agentId: "main",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      channelId: "telegram",
      sessionFile: undefined,
      staging: [
        "A new session was started via /new or /reset. Execute your Session Startup sequence now - read the required files before responding to the user.",
        "我剛才跨 agent smoke token 是 bridge-8841 嗎？",
      ],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const result = await engine.assemble({
      sessionId: "startup-filter",
      messages: [
        {
          role: "user",
          content: "A new session was started via /new or /reset. Execute your Session Startup sequence now - read the required files before responding to the user.",
        },
      ],
    });

    assert.match(result.systemPromptAddition, /bridge-8841/);
    assert.strictEqual(deps.diagnostics.lastAssemble?.query.includes("bridge-8841"), true);
    assert.strictEqual(deps.diagnostics.lastAssemble?.query.includes("/new or /reset"), false);
  } finally {
    await safeRm(dbPath);
  }
});

test("Phase 3: assemble ignores literal session bootstrap placeholder and falls back to staged user query", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    await insertMemory(storage, {
      title: "Yesterday marker",
      content: "昨天的重要代號是 bridge-8841",
      summary: "bridge-8841",
      memory_scope: "owner_shared",
      embedding: await createFakeEmbedder().embed("bridge-8841"),
    });

    const deps = createDeps(storage);
    deps.sessionStates.set("literal-bootstrap-filter", {
      sessionId: "literal-bootstrap-filter",
      sessionKey: "key-literal-bootstrap-filter",
      agentId: "main",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      messageChannel: "telegram",
      requesterSenderId: "0000000000",
      agentAccountId: "bot-account",
      senderIsOwner: true,
      channelId: "telegram",
      sessionFile: undefined,
      staging: [
        "(session bootstrap)",
        "你記得昨天的事嗎 bridge-8841",
      ],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const result = await engine.assemble({
      sessionId: "literal-bootstrap-filter",
      messages: [{ role: "user", content: "(session bootstrap)" }],
    });

    assert.match(result.systemPromptAddition, /bridge-8841/);
    assert.strictEqual(deps.diagnostics.lastAssemble?.query.includes("bridge-8841"), true);
    assert.strictEqual(deps.diagnostics.lastAssemble?.query.includes("(session bootstrap)"), false);
  } finally {
    await safeRm(dbPath);
  }
});

test("Phase 3: memory-intent query with no hits injects no-reset contract", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    const deps = createDeps(storage);
    deps.sessionStates.set("no-hit-memory-intent", {
      sessionId: "no-hit-memory-intent",
      sessionKey: "key-no-hit-memory-intent",
      agentId: "main",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      messageChannel: "telegram",
      requesterSenderId: "0000000000",
      agentAccountId: "bot-account",
      senderIsOwner: true,
      channelId: "telegram",
      sessionFile: undefined,
      staging: [],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const result = await engine.assemble({
      sessionId: "no-hit-memory-intent",
      messages: [{ role: "user", content: "你記得昨天的事嗎" }],
    });

    assert.match(result.systemPromptAddition, /No relevant long-term memory was retrieved for this query/);
    assert.match(result.systemPromptAddition, /Do NOT claim that a restart/);
    assert.match(
      deps.sessionStates.get("no-hit-memory-intent")?.pendingPromptContext ?? "",
      /No relevant long-term memory was found for this query/,
    );
  } finally {
    await safeRm(dbPath);
  }
});

test("Phase 3: buildRecallQuery uses the latest user turn instead of concatenating prior turns", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    await insertMemory(storage, {
      title: "Yesterday marker",
      content: "昨天完成 memory cutover 並停用 session-memory hook",
      summary: "昨天完成 memory cutover",
      memory_scope: "owner_shared",
      embedding: await createFakeEmbedder().embed("你記得昨天的事嗎 memory cutover"),
    });

    const deps = createDeps(storage);
    deps.sessionStates.set("latest-user-turn", {
      sessionId: "latest-user-turn",
      sessionKey: "key-latest-user-turn",
      agentId: "main",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      messageChannel: "telegram",
      requesterSenderId: "0000000000",
      agentAccountId: "bot-account",
      senderIsOwner: true,
      channelId: "telegram",
      sessionFile: undefined,
      staging: [],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const result = await engine.assemble({
      sessionId: "latest-user-turn",
      messages: [
        { role: "user", content: "[Fri 2026-03-20 15:34 GMT+8] 你好" },
        { role: "assistant", content: "你好，我是 plaw。今天想處理什麼？" },
        {
          role: "user",
          content: "Conversation info (untrusted metadata):\n```json\n{\"message_id\":\"2321\"}\n```\n\n你記得昨天的事嗎",
        },
      ],
    });

    assert.match(result.systemPromptAddition, /昨天完成 memory cutover/);
    assert.match(result.systemPromptAddition, /The user is explicitly asking about prior discussions or remembered facts/);
    assert.match(result.systemPromptAddition, /Do NOT say that you do not remember/);
    assert.strictEqual(deps.diagnostics.lastAssemble?.query, "你記得昨天的事嗎");
  } finally {
    await safeRm(dbPath);
  }
});

test("Phase 3: assemble uses pendingPrompt when the current turn is not yet in session messages", async () => {
  const { dbPath, storage } = await setupStorage();
  try {
    await insertMemory(storage, {
      title: "Yesterday marker",
      content: "昨天完成 memory cutover 並停用 session-memory hook",
      summary: "昨天完成 memory cutover",
      memory_scope: "owner_shared",
      embedding: await createFakeEmbedder().embed("你記得昨天的事嗎 memory cutover"),
    });

    const deps = createDeps(storage);
    deps.sessionStates.set("pending-prompt-query", {
      sessionId: "pending-prompt-query",
      sessionKey: "key-pending-prompt-query",
      agentId: "main",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      messageChannel: "telegram",
      requesterSenderId: "0000000000",
      agentAccountId: "bot-account",
      senderIsOwner: true,
      channelId: "telegram",
      pendingPrompt:
        "Conversation info (untrusted metadata):\n```json\n{\"message_id\":\"2326\"}\n```\n\n你記得昨天的事嗎",
      sessionFile: undefined,
      staging: ["你好，我是 plaw。今天想處理什麼？"],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });

    const engine = createMemoryBrainContextEngine(deps);
    const result = await engine.assemble({
      sessionId: "pending-prompt-query",
      messages: [{ role: "assistant", content: "你好，我是 plaw。今天想處理什麼？" }],
    });

    assert.match(result.systemPromptAddition, /昨天完成 memory cutover/);
    assert.strictEqual(deps.diagnostics.lastAssemble?.query, "你記得昨天的事嗎");
    assert.match(
      deps.sessionStates.get("pending-prompt-query")?.pendingPromptContext ?? "",
      /Use the recalled memories below directly if they answer the question/,
    );
    assert.match(
      deps.sessionStates.get("pending-prompt-query")?.pendingPromptContext ?? "",
      /昨天完成 memory cutover 並停用 session-memory hook/,
    );
  } finally {
    await safeRm(dbPath);
  }
});

test("Phase 3: import-legacy sanitizes markdown and dedupes unchanged reruns", async () => {
  const { dbPath, storage } = await setupStorage();
  const legacyDir = `/tmp/memory-lancedb-brain-legacy-${randomUUID()}`;
  try {
    await mkdir(legacyDir, { recursive: true });
    await writeFile(
      `${legacyDir}/2026-03-19-note.md`,
      [
        "# 昨天的部署",
        "",
        "Conversation info (untrusted metadata):",
        "```json",
        "{\"message_id\":\"1\"}",
        "```",
        "",
        "昨天完成 brain migration，並決定停用 session-memory。",
      ].join("\n"),
      "utf8",
    );
    await writeFile(
      `${legacyDir}/favorite.md`,
      [
        "# 喜好",
        "",
        "用戶偏好直接、簡潔的回答。",
      ].join("\n"),
      "utf8",
    );

    const deps = createDeps(storage);
    deps.sessionStates.set("legacy-import", {
      sessionId: "legacy-import",
      sessionKey: "key-legacy-import",
      agentId: "main",
      owner: { ownerId: "test-user", ownerNamespace: "personal" },
      messageChannel: "telegram",
      requesterSenderId: "0000000000",
      agentAccountId: "bot-account",
      senderIsOwner: true,
      channelId: "telegram",
      sessionFile: undefined,
      staging: [],
      childSessionKeys: [],
      updatedAt: Date.now(),
    });
    deps.lastSessionByChannel.set("telegram", "legacy-import");

    const engine = createMemoryBrainContextEngine(deps);
    const command = createMemoryDistillCommand(deps, engine);
    const firstRun = await command.handler({ args: `import-legacy ${legacyDir}`, channel: "telegram" });
    assert.match(firstRun.text, /Imported: 2/);
    assert.match(firstRun.text, /Updated: 0/);

    const imported = await storage.queryMemoriesByFilter({ source: "ingest:legacy-markdown", status: "active" });
    assert.strictEqual(imported.length, 2);
    assert.ok(imported.some((memory) => memory.content.includes("昨天完成 brain migration")));
    assert.ok(imported.every((memory) => !memory.content.includes("Conversation info (untrusted metadata)")));

    const secondRun = await command.handler({ args: `migrate-legacy ${legacyDir}`, channel: "telegram" });
    assert.match(secondRun.text, /Skipped: 2/);

    await writeFile(
      `${legacyDir}/favorite.md`,
      [
        "# 喜好",
        "",
        "用戶偏好直接、簡潔的回答，而且結論先行。",
      ].join("\n"),
      "utf8",
    );

    const thirdRun = await command.handler({ args: `import-legacy ${legacyDir}`, channel: "telegram" });
    assert.match(thirdRun.text, /Updated: 1/);

    const activeImported = await storage.queryMemoriesByFilter({ source: "ingest:legacy-markdown", status: "active" });
    assert.strictEqual(activeImported.length, 2);
    assert.ok(activeImported.some((memory) => memory.content.includes("結論先行")));
  } finally {
    await safeRm(dbPath);
    await safeRm(legacyDir);
  }
});
