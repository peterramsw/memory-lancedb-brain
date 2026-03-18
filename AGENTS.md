# AGENTS.md — Review Guidelines for memory-lancedb-brain

## What this project is

An OpenClaw context-engine plugin that provides persistent, owner-scoped long-term memory backed by LanceDB. It distills conversations into structured memories and injects them into future prompts via hybrid retrieval.

## Architecture invariants

1. **Fail-closed writes** — All memory write paths (`compactSession`, `/memory store`, distillation) refuse to proceed without a resolved `owner_id` + `owner_namespace`. Never weaken this guard.
2. **Owner resolution chain** — Channel match → any-channel match → senderIsOwner (single owner) → first configured owner. When `owners` is configured, never synthesize a "default" namespace owner from senderId/agentId.
3. **Session file is the source of truth** — JSONL format with `session`, `message`, `compaction`, `model_change` entry types. Truncation in `compact()` must preserve the header entry and prepend a `compaction` marker.
4. **`ownsCompaction: false`** — The plugin does NOT own the message compaction lifecycle. OpenClaw's Pi handles in-streaming compaction. The plugin's `compact()` is called only on token-overflow recovery.

## Key files

| File | Role |
|------|------|
| `index.ts` | Plugin entry point, event handlers, global state |
| `src/context-engine.ts` | Context engine hooks (bootstrap/afterTurn/assemble/compact) |
| `src/owners.ts` | Owner resolution and access control |
| `src/retrieval.ts` | Hybrid search (vector + keyword + rerank blending) |
| `src/distill.ts` | LLM distillation + heuristic fallback |
| `src/lifecycle.ts` | Merge, supersede, archive pipeline |
| `src/storage.ts` | LanceDB CRUD + vector search |
| `openclaw.plugin.json` | Plugin manifest and config schema |

## Review checklist

- [ ] Does the change preserve fail-closed write semantics?
- [ ] Does owner resolution still follow the documented chain?
- [ ] Are new config fields reflected in both `openclaw.plugin.json` schema and `README.md`?
- [ ] Does `compact()` still truncate the session file (not just extract memories)?
- [ ] Are there no new `"default"` namespace synthesizations when `owners` is configured?
