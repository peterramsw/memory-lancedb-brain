# memory-lancedb-brain

OpenClaw plugin for persistent, intelligent long-term memory backed by [LanceDB](https://lancedb.com/).

Conversations are automatically distilled into structured memories (facts, preferences, decisions, etc.) using an LLM, stored with vector embeddings, and injected into future prompts via semantic search — giving agents cross-session recall without manual `/memory` commands.

## Features

- **LLM-powered distillation** — Extracts user-stated facts, preferences, and decisions from conversation transcripts (not raw transcript fragments)
- **Hybrid retrieval** — Vector similarity + keyword matching with optional cross-encoder reranking
- **Lifecycle management** — Automatic merge (>0.92 similarity), supersede detection (>0.80), and cold-memory archival
- **Multi-owner isolation** — Owner-scoped memories shared across agents, plus agent-local memories
- **Context engine integration** — Registers as an openclaw context engine; memories are automatically assembled into system prompts
- **Auto-distill** — Triggers on session end, `/new`, `/reset`, or subagent completion

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
  "channels": { "telegram:123456": "123456" }
}]'
```

### 4. Restart gateway

```bash
systemctl --user restart openclaw-gateway.service
```

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
  "channels": { "telegram:123456": "123456" }
}]
```

The `channels` map binds incoming messages to owners. Each key uses the format `<channel>:<senderId>` (e.g., `"telegram:123456"`, `"line:U1a2b3c"`), where `<channel>` matches the OpenClaw `messageChannel` and `<senderId>` matches the sender's platform ID. The value is an arbitrary label (typically the same sender ID).

When a message arrives, the plugin resolves the owner by matching `messageChannel` + `senderId` against these bindings. If no channel match is found but `senderIsOwner` is true and there is exactly one owner, that owner is used. As a final fallback (e.g., during compaction where no runtime context is available), the first configured owner is used.

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
```

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
  }
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

## Self-Improving Agent

The plugin automatically learns from mistakes and corrections:

1. **User corrects the agent** ("不對，應該用 pnpm") → distilled as `correction` type
2. **User confirms an approach works** ("這招有效") → distilled as `best_practice` type
3. **Same correction/practice appears again** → merge lifecycle bumps `importance` (+1 each merge, max 5)
4. **High-importance memories rank first** in retrieval via recency + importance scoring

No manual promotion, no file management, no grep. The vector similarity merge IS the recurrence counter — semantically similar learnings get merged and importance grows automatically.

## License

MIT
