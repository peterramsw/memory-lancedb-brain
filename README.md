# memory-lancedb-brain

OpenClaw plugin for persistent, intelligent long-term memory backed by [LanceDB](https://lancedb.com/).

Conversations are automatically distilled into structured memories (facts, preferences, decisions, etc.) using an LLM, stored with vector embeddings, and injected into future prompts via semantic search — giving agents cross-session and cross-agent recall without manual `/memory` commands.

Distillation is best-effort rather than brittle: startup/session boilerplate is sanitized before compaction, transcript size is bounded by `autoDistill.tokenBudget`, and heuristic fallback is designed to preserve useful user-stated facts when the LLM distiller path fails.

## Features

- **LLM-powered distillation** — Extracts user-stated facts, preferences, and decisions from conversation transcripts (not raw transcript fragments)
- **Hybrid retrieval** — Vector similarity + keyword matching with optional cross-encoder reranking
- **Lifecycle management** — Automatic merge (>0.92 similarity), supersede detection (>0.80), and cold-memory archival
- **Multi-owner isolation** — Owner-scoped memories shared across agents, plus agent-local memories
- **Context engine integration** — Registers as an openclaw context engine; memories are automatically assembled into system prompts
- **Auto-distill** — Triggers on session end, `/new`, `/reset`, or subagent completion
- **Transcript hygiene + bounded distill budget** — Compact/distill strips startup boilerplate, untrusted metadata, and command noise before sending the transcript to the distiller
- **Fallback fact preservation** — If LLM distillation fails, heuristic fallback still keeps user-stated operational facts and future plans instead of persisting startup/session boilerplate
- **Status affordance** — `memory_status` reports loaded state, DB path, recent assemble/compact evidence, and explains that brain runs in-process inside `openclaw-gateway`
- **Legacy markdown migration** — `/memory import-legacy [path]` imports existing markdown memory notes into LanceDB with sanitization and rerun-safe dedupe
- **No-reset memory contract** — For memory-intent queries with no hit, brain injects a contract that says no relevant long-term memory was found, instead of implying all memory was wiped by `/new` or restart

## Architecture

```
index.ts                 Plugin entry point, event handlers, global state
src/
  context-engine.ts      Context engine (bootstrap/ingest/assemble/compact)
  distill.ts             LLM distillation + heuristic fallback
  lifecycle.ts           Merge, supersede, archive pipeline
  retrieval.ts           Hybrid search (vector + keyword + rerank)
  storage.ts             LanceDB CRUD + vector search
  embedding.ts           OpenAI-compatible embedding client
  schema.ts              MemoryRecord, MemoryEventRecord types
  tools.ts               Memory management tools
  owners.ts              Owner resolution and multi-tenant config
```

## Installation

### 1. Clone and install dependencies

```bash
git clone <repo-url> /path/to/memory-lancedb-brain
cd /path/to/memory-lancedb-brain
npm install
```

### 2. Register in openclaw config

```bash
# Add plugin load path
openclaw config set plugins.load.paths '["<path-to>/memory-lancedb-brain"]'

# Enable the plugin
openclaw config set plugins.entries.memory-lancedb-brain.enabled true

# Activate as context engine (required for auto-assemble + auto-distill)
openclaw config set plugins.slots.contextEngine memory-lancedb-brain
```

### 3. Configure (minimal)

```bash
# Set embedding provider (any OpenAI-compatible API)
openclaw config set plugins.entries.memory-lancedb-brain.config.embedding '{
  "apiKey": "sk-...",
  "model": "text-embedding-3-small",
  "baseURL": "https://api.openai.com/v1",
  "dimensions": 1536
}'

# Set distillation model (any chat completions API)
openclaw config set plugins.entries.memory-lancedb-brain.config.distillation '{
  "model": "gpt-4o-mini",
  "baseURL": "https://api.openai.com/v1",
  "apiKey": "sk-..."
}'

# Set owner(s) — required for memory isolation
openclaw config set plugins.entries.memory-lancedb-brain.config.owners '[{
  "owner_id": "peter",
  "owner_namespace": "owner_shared",
  "channels": { "telegram": "123456" }
}]'
```

### 4. Restart gateway

```bash
systemctl --user restart openclaw-gateway.service
```

### 5. Migrate existing markdown memory notes (recommended for post-install adoption)

If this OpenClaw workspace already has legacy markdown memory files, import them once after installing brain:

```bash
/memory import-legacy ~/.openclaw/workspace/memory
```

You can also import a single file:

```bash
/memory import-legacy ~/.openclaw/workspace/MEMORY.md
```

Rerunning the same import is safe: unchanged files are skipped, and changed files supersede the previous imported version instead of creating duplicate active memories.

## Configuration Reference

All configuration is set via `plugins.entries.memory-lancedb-brain.config`:

### `embedding` (required)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `apiKey` | string | `$OPENCLAW_EMBEDDING_API_KEY` or `"local"` | API key for embedding service |
| `model` | string | `$OPENCLAW_EMBEDDING_MODEL` or `"text-embedding-3-small"` | Embedding model name |
| `baseURL` | string | `$OPENCLAW_EMBEDDING_BASE_URL` or `"https://api.openai.com/v1"` | Embedding API base URL |
| `dimensions` | number | `$OPENCLAW_EMBEDDING_DIMENSIONS` or `1536` | Embedding vector dimensions |

### `distillation` (required for auto-distill)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `$OPENCLAW_DISTILL_MODEL` or `"gpt-4o-mini"` | Chat model for transcript distillation |
| `baseURL` | string | `$OPENCLAW_DISTILL_BASE_URL` or `"https://api.openai.com/v1"` | Chat completions API base URL |
| `apiKey` | string | `$OPENCLAW_DISTILL_API_KEY` | API key for distillation |

### `retrieval` (optional)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"hybrid"` \| `"vector"` \| `"keyword"` | `"hybrid"` | Retrieval strategy |
| `vectorWeight` | number | — | Weight for vector similarity in hybrid mode |
| `bm25Weight` | number | — | Weight for keyword matching in hybrid mode |
| `minScore` | number | — | Minimum relevance score to include |
| `hardMinScore` | number | — | Hard cutoff below which results are dropped |
| `rerank` | boolean | `false` | Enable cross-encoder reranking |
| `rerankModel` | string | — | Reranker model name |
| `rerankEndpoint` | string | — | Reranker API endpoint URL |
| `rerankApiKey` | string | — | Reranker API key |
| `candidatePoolSize` | number | — | Number of candidates to fetch before reranking |

#### Rerank blending formula

When `rerank: true`, the retrieval pipeline first fetches `candidatePoolSize` candidates via hybrid search, then sends them to the cross-encoder reranker. The final score for each candidate is computed as:

- **Reranked candidate**: `rerankScore × 0.6 + originalScore × 0.4`
- **Unmatched candidate** (not returned by reranker): `originalScore × 0.8`

This blending ensures that the reranker can promote or demote results while the original hybrid score still contributes as a stabilizing signal. Results below `hardMinScore` are dropped after blending.

### `autoDistill` (optional)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable automatic distillation |
| `triggers` | string[] | `["onSubagentEnded", "onSessionEnd", "onReset", "onNew"]` | Events that trigger distillation |
| `minStagingLength` | number | `3` | Minimum staged messages before distilling |
| `tokenBudget` | number | `30000` | Max transcript tokens to send to distiller |

### `owners` (required)

Array of owner definitions for multi-tenant memory isolation. **All memory write operations are fail-closed** — without a valid `owner_id` and `owner_namespace`, distillation, `/memory store`, and compaction will refuse to write. This is a security invariant: memories must always be attributable to a configured owner.

```json
[{
  "owner_id": "user-1",
  "owner_namespace": "owner_shared",
  "channels": { "telegram": "123456" }
}]
```

The `channels` map binds incoming messages to owners. Each key is the **channel name** (e.g., `"telegram"`, `"line"`), matching the OpenClaw `messageChannel` field. The value is the **sender's platform ID** (e.g., `"123456"`, `"U1a2b3c"`). The plugin resolves the owner by looking up `owner.channels[messageChannel]` and comparing the value against `senderId`.

When a message arrives, if no channel match is found but `senderIsOwner` is true and there is exactly one owner, that owner is used. As a final fallback (e.g., during compaction where no runtime context is available), the first configured owner is used.

### `dbPath` (optional)

LanceDB storage directory. Default: `./data/memory-lancedb-brain` (relative to plugin directory).

### `agentWhitelist` (optional)

Array of agent IDs allowed to use memory tools. Default: `["main"]`.

## Usage

### Automatic (recommended)

Once configured, the plugin works automatically:

1. **Conversations happen** on any channel (Telegram, LINE, etc.)
2. **`assemble()`** injects relevant memories into the system prompt before each agent turn
3. **`session_end`** triggers LLM distillation of the conversation into structured memories
4. **Next conversation** — agent sees relevant past memories in its context

### Manual commands

```
/memory distill          Force-distill current session to LanceDB
/memory recall <query>   Search memories by semantic query
/memory list [scope]     List memories (owner_shared / agent_local / all)
/memory store <text>     Store a single memory immediately
/memory import-legacy [path]  Import legacy markdown notes into LanceDB
/memory migrate-legacy [path] Alias of import-legacy
```

### Agent tools

The plugin exposes seven tools:

- `memory_recall`
- `memory_store`
- `memory_forget`
- `memory_update`
- `memory_list`
- `memory_stats`
- `memory_status`

`memory_status` is the canonical operator/agent health check. It reports whether the plugin is loaded, which owner/scopes are active for the current runtime context, recent assemble/afterTurn/compact evidence, and a reminder that no standalone LanceDB process/container is expected because brain is an in-process OpenClaw plugin.

## Migration Notes

`memory-lancedb-brain` is designed for the common case where users install brain after already using OpenClaw for a while.

- Legacy markdown notes can be imported from a whole directory or a single `.md` file.
- Known delivery noise such as `Conversation info (untrusted metadata)` and startup boilerplate is stripped before import.
- Imported records are stored with `source = "ingest:legacy-markdown"`.
- Re-running import against unchanged files skips them; changed files supersede the previous imported version.

## Example: Local vLLM Setup

For self-hosted inference (e.g., with vLLM behind a reverse proxy):

```json
{
  "embedding": {
    "apiKey": "local",
    "model": "vllm/Forturne/Qwen3-Embedding-4B-NVFP4",
    "baseURL": "http://127.0.0.1:32080/v1",
    "dimensions": 2560
  },
  "distillation": {
    "model": "vllm/Sehyo/Qwen3.5-35B-A3B-NVFP4",
    "baseURL": "http://127.0.0.1:32080/v1",
    "apiKey": "local"
  },
  "retrieval": {
    "mode": "hybrid",
    "rerank": true,
    "rerankModel": "vllm/Forturne/Qwen3-Reranker-4B-NVFP4",
    "rerankEndpoint": "http://127.0.0.1:32080/v1/rerank",
    "rerankApiKey": "local"
  },
  "owners": [{
    "owner_id": "peter",
    "owner_namespace": "owner_shared",
    "channels": { "telegram": "123456" }
  }]
}
```

## Memory Types

| Type | Scope | Description |
|------|-------|-------------|
| `fact` | owner_shared | Personal facts about the user |
| `preference` | owner_shared | Likes, dislikes, habits, routines |
| `decision` | owner_shared | Choices the user made |
| `summary` | owner_shared | Session summaries |
| `pitfall` | agent_local | Problems or bugs encountered |
| `todo` | agent_local | Unfinished tasks |
| `correction` | owner_shared | User corrections — "not X, should be Y" |
| `best_practice` | owner_shared | Proven approaches — "use Y when doing X" |

## Memory Source Tracking

Every memory record includes a `source` field that tracks how it was created. This is useful for auditing, filtering, and future data migration.

| Source | Description |
|--------|-------------|
| `distill` | Automatically extracted from conversation via LLM distillation |
| `manual` | Explicitly stored via `/memory store` command or `memory_store` tool |
| `consolidate` | Generated by the consolidation lifecycle (merging related fragments) |
| `synthesize` | Generated by user profile synthesis |
| `unknown` | Legacy records created before source tracking was added |
| `ingest:*` | Reserved for external data source ingestion (e.g., `ingest:graph-email`) |

Existing databases are automatically migrated — old records receive `source: "unknown"` when the plugin starts. No manual intervention is needed.

## Self-Improving Agent

The plugin automatically learns from mistakes and corrections:

1. **User corrects the agent** ("不對，應該用 pnpm") → distilled as `correction` type
2. **User confirms an approach works** ("這招有效") → distilled as `best_practice` type
3. **Same correction/practice appears again** → merge lifecycle bumps `importance` (+1 each merge, max 5)
4. **High-importance memories rank first** in retrieval via recency + importance scoring

No manual promotion, no file management, no grep. The vector similarity merge IS the recurrence counter — semantically similar learnings get merged and importance grows automatically.

## License

MIT
