import { createHash, randomUUID } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import { basename, extname, relative, resolve } from "node:path";
import type { Embedder } from "./embedding.js";
import type { MemoryEventRecord, MemoryRecord, MemoryType } from "./schema.js";
import type { MemoryStorage } from "./storage.js";
import { detectCategory, generateSummary } from "./retrieval.js";

export const LEGACY_IMPORT_SOURCE = "ingest:legacy-markdown";
const LEGACY_PATH_PREFIX = "legacy-path:";
const LEGACY_HASH_PREFIX = "legacy-sha256:";

export interface LegacyImportDeps {
  storage: MemoryStorage;
  embedder: Embedder;
}

export interface LegacyImportParams {
  rootPath: string;
  ownerId: string;
  ownerNamespace: string;
  agentId: string;
  sourceSessionId: string;
}

export interface LegacyImportResult {
  rootPath: string;
  filesDiscovered: number;
  filesConsidered: number;
  imported: number;
  updated: number;
  skipped: number;
  errors: string[];
}

type ExistingLegacyMemory = {
  record: MemoryRecord;
  legacyPath?: string;
  legacyHash?: string;
};

function collapseWhitespace(text: string): string {
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]+\n/g, "\n")
    .trim();
}

export function sanitizeLegacyMarkdown(raw: string): string {
  let text = raw.replace(/\r\n/g, "\n");
  text = text.replace(/Conversation info \(untrusted metadata\):\s*```json[\s\S]*?```\s*/gi, "");
  text = text.replace(/Sender \(untrusted metadata\):\s*```json[\s\S]*?```\s*/gi, "");
  text = text.replace(/A new session was started via \/new or \/reset[\s\S]*?(?:Current time:[^\n]*\n?)?/gi, "");
  text = text.replace(/^\(session bootstrap\)\s*$/gim, "");
  text = text.replace(/^\[[^\]]*GMT[^\]]*\]\s*/gm, "");
  text = text.replace(/^\s*Current time:[^\n]*$/gim, "");
  return collapseWhitespace(text);
}

function parseTags(tags: string): string[] {
  try {
    const parsed = JSON.parse(tags);
    return Array.isArray(parsed) ? parsed.map((item) => String(item)) : [];
  } catch {
    return [];
  }
}

function deriveLegacyTitle(filePath: string, sanitized: string): string {
  const heading = sanitized
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line.startsWith("#"));
  if (heading) {
    return heading.replace(/^#+\s*/, "").trim().slice(0, 120);
  }
  return basename(filePath, extname(filePath)).replace(/[-_]+/g, " ").trim().slice(0, 120);
}

function chooseMemoryType(title: string, content: string): MemoryType {
  const detected = detectCategory(`${title}\n${content}`);
  if (detected === "status") return "summary";
  return detected;
}

function buildLegacyTags(relPath: string, hash: string): string {
  return JSON.stringify([
    "legacy-import",
    `${LEGACY_PATH_PREFIX}${relPath}`,
    `${LEGACY_HASH_PREFIX}${hash}`,
  ]);
}

function extractLegacyMeta(record: MemoryRecord): ExistingLegacyMemory {
  const tags = parseTags(record.tags);
  return {
    record,
    legacyPath: tags.find((tag) => tag.startsWith(LEGACY_PATH_PREFIX))?.slice(LEGACY_PATH_PREFIX.length),
    legacyHash: tags.find((tag) => tag.startsWith(LEGACY_HASH_PREFIX))?.slice(LEGACY_HASH_PREFIX.length),
  };
}

async function collectMarkdownFiles(rootPath: string): Promise<string[]> {
  const info = await stat(rootPath);
  if (info.isFile()) {
    return rootPath.toLowerCase().endsWith(".md") ? [rootPath] : [];
  }

  const results: string[] = [];
  const entries = await readdir(rootPath, { withFileTypes: true });
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const nextPath = resolve(rootPath, entry.name);
    if (entry.isDirectory()) {
      results.push(...await collectMarkdownFiles(nextPath));
      continue;
    }
    if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
      results.push(nextPath);
    }
  }
  return results;
}

function computeHash(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 16);
}

function buildContentForEmbedding(title: string, sanitized: string): string {
  return `${title}\n${sanitized}`.slice(0, 1800);
}

export async function importLegacyMarkdown(
  deps: LegacyImportDeps,
  params: LegacyImportParams,
): Promise<LegacyImportResult> {
  const rootPath = resolve(params.rootPath);
  const files = await collectMarkdownFiles(rootPath);
  const existing = await deps.storage.queryMemoriesByFilter({
    owner_id: params.ownerId,
    owner_namespace: params.ownerNamespace,
    source: LEGACY_IMPORT_SOURCE,
    status: "active",
  });
  const existingByPath = new Map<string, ExistingLegacyMemory>();
  for (const record of existing) {
    const meta = extractLegacyMeta(record);
    if (meta.legacyPath) existingByPath.set(meta.legacyPath, meta);
  }

  const result: LegacyImportResult = {
    rootPath,
    filesDiscovered: files.length,
    filesConsidered: 0,
    imported: 0,
    updated: 0,
    skipped: 0,
    errors: [],
  };

  for (const filePath of files) {
    try {
      const raw = await readFile(filePath, "utf8");
      const sanitized = sanitizeLegacyMarkdown(raw);
      if (sanitized.length < 12) {
        result.skipped += 1;
        continue;
      }

      result.filesConsidered += 1;
      const relPath = files.length === 1 ? basename(filePath) : relative(rootPath, filePath);
      const hash = computeHash(sanitized);
      const existingMatch = existingByPath.get(relPath);
      if (existingMatch?.legacyHash === hash) {
        result.skipped += 1;
        continue;
      }

      const title = deriveLegacyTitle(filePath, sanitized);
      const content = sanitized.slice(0, 4000);
      const embedding = await deps.embedder.embed(buildContentForEmbedding(title, content));
      const now = Date.now();
      const memoryId = randomUUID();

      if (existingMatch) {
        await deps.storage.updateMemory(existingMatch.record.memory_id, {
          status: "superseded",
          last_used_at: now,
        });
      }

      const record: MemoryRecord = {
        memory_id: memoryId,
        owner_namespace: params.ownerNamespace,
        owner_id: params.ownerId,
        agent_id: params.agentId,
        memory_scope: "owner_shared",
        memory_type: chooseMemoryType(title, content),
        title,
        content,
        summary: generateSummary(content),
        tags: buildLegacyTags(relPath, hash),
        importance: 3,
        confidence: 0.85,
        status: "active",
        supersedes_id: existingMatch?.record.memory_id ?? "",
        created_at: now,
        updated_at: now,
        last_used_at: now,
        source_session_id: params.sourceSessionId,
        source: LEGACY_IMPORT_SOURCE,
        embedding,
      };

      await deps.storage.insertMemory(record);
      const event: MemoryEventRecord = {
        event_id: randomUUID(),
        memory_id: memoryId,
        event_type: "create",
        event_time: now,
        details_json: JSON.stringify({
          source_path: relPath,
          source_hash: hash,
          imported_from: rootPath,
          import_mode: "legacy-markdown",
          superseded_memory_id: existingMatch?.record.memory_id ?? "",
        }),
      };
      await deps.storage.insertEvent(event);

      if (existingMatch) result.updated += 1;
      else result.imported += 1;
    } catch (error) {
      result.errors.push(`${filePath}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  return result;
}
