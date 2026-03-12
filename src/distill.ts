/**
 * Distillation support for memory-lancedb-brain
 */

export interface DistillOutput {
  session_summary: string;
  confirmed_facts: string[];
  decisions: string[];
  pitfalls: string[];
  preference_updates: string[];
  environment_truths: string[];
  open_loops: string[];
  scope_recommendation: "owner_shared" | "agent_local" | "both";
}

export interface DistillerConfig {
  model?: string;
  baseURL?: string;
  apiKey?: string;
  fetchImpl?: typeof fetch;
  distillImpl?: (transcript: string, opts?: { customInstructions?: string }) => Promise<DistillOutput>;
}

const DEFAULT_OUTPUT: DistillOutput = {
  session_summary: "",
  confirmed_facts: [],
  decisions: [],
  pitfalls: [],
  preference_updates: [],
  environment_truths: [],
  open_loops: [],
  scope_recommendation: "agent_local",
};

function ensureArrayOfStrings(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item ?? "").trim()).filter(Boolean);
}

function normalizeOutput(payload: Partial<DistillOutput> | null | undefined): DistillOutput {
  const scope = payload?.scope_recommendation;
  return {
    session_summary: String(payload?.session_summary ?? "").trim(),
    confirmed_facts: ensureArrayOfStrings(payload?.confirmed_facts),
    decisions: ensureArrayOfStrings(payload?.decisions),
    pitfalls: ensureArrayOfStrings(payload?.pitfalls),
    preference_updates: ensureArrayOfStrings(payload?.preference_updates),
    environment_truths: ensureArrayOfStrings(payload?.environment_truths),
    open_loops: ensureArrayOfStrings(payload?.open_loops),
    scope_recommendation:
      scope === "owner_shared" || scope === "agent_local" || scope === "both"
        ? scope
        : "agent_local",
  };
}

export function parseDistillJson(text: string): DistillOutput | null {
  const trimmed = text.trim();
  const candidates = [trimmed];

  const codeBlockMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (codeBlockMatch?.[1]) candidates.push(codeBlockMatch[1].trim());

  const jsonObjectMatch = trimmed.match(/\{[\s\S]*\}/);
  if (jsonObjectMatch?.[0]) candidates.push(jsonObjectMatch[0]);

  for (const candidate of candidates) {
    try {
      return normalizeOutput(JSON.parse(candidate));
    } catch {
      // continue
    }
  }

  return null;
}

export function heuristicDistill(transcript: string): DistillOutput {
  const lines = transcript
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-20);

  const summary = lines.slice(-6).join(" ").slice(0, 500);
  const facts = lines.filter((line) => /is |are |使用|在|會|runs|uses/i.test(line)).slice(0, 3);
  const decisions = lines.filter((line) => /決定|改用|decide|選擇|will use/i.test(line)).slice(0, 3);
  const pitfalls = lines.filter((line) => /不要|避免|坑|error|fail|bug/i.test(line)).slice(0, 3);
  const preferences = lines.filter((line) => /偏好|喜歡|prefer/i.test(line)).slice(0, 3);
  const todos = lines.filter((line) => /待辦|todo|接下來|next|需要/i.test(line)).slice(0, 3);

  return normalizeOutput({
    ...DEFAULT_OUTPUT,
    session_summary: summary,
    confirmed_facts: facts,
    decisions,
    pitfalls,
    preference_updates: preferences,
    environment_truths: facts,
    open_loops: todos,
    scope_recommendation: preferences.length > 0 || facts.length > 0 ? "owner_shared" : "agent_local",
  });
}

async function callChatCompletions(config: Required<Pick<DistillerConfig, "baseURL" | "model" | "apiKey">> & DistillerConfig, transcript: string, customInstructions?: string): Promise<string> {
  const fetchImpl = config.fetchImpl ?? fetch;
  const response = await fetchImpl(`${config.baseURL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model: config.model,
      temperature: 0.1,
      messages: [
        {
          role: "system",
          content:
            "You are a memory distillation engine. Return valid JSON only. No markdown, no commentary. The JSON schema is: { session_summary: string, confirmed_facts: string[], decisions: string[], pitfalls: string[], preference_updates: string[], environment_truths: string[], open_loops: string[], scope_recommendation: 'owner_shared' | 'agent_local' | 'both' }",
        },
        {
          role: "user",
          content:
            `${customInstructions ? `${customInstructions}\n\n` : ""}Distill the following transcript into structured memory JSON:\n\n${transcript}`,
        },
      ],
    }),
  });

  if (!response.ok) {
    throw new Error(`Distillation API error: ${response.status} ${response.statusText}`);
  }

  const payload = await response.json();
  return String(payload?.choices?.[0]?.message?.content ?? "");
}

export function createDistiller(config: DistillerConfig) {
  const resolved = {
    model: config.model ?? "vllm/Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
    baseURL: config.baseURL ?? "http://127.0.0.1:32080/v1",
    apiKey: config.apiKey ?? "local",
    ...config,
  };

  return {
    async distillTranscript(transcript: string, opts?: { customInstructions?: string }): Promise<DistillOutput> {
      if (resolved.distillImpl) {
        return normalizeOutput(await resolved.distillImpl(transcript, opts));
      }

      try {
        const raw = await callChatCompletions(resolved, transcript, opts?.customInstructions);
        return parseDistillJson(raw) ?? heuristicDistill(transcript);
      } catch {
        return heuristicDistill(transcript);
      }
    },
  };
}
