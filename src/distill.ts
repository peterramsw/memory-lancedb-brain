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
  corrections: string[];
  best_practices: string[];
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
  corrections: [],
  best_practices: [],
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
    corrections: ensureArrayOfStrings(payload?.corrections),
    best_practices: ensureArrayOfStrings(payload?.best_practices),
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
    .filter(Boolean);

  // Only extract from USER lines, not assistant lines
  const userLines = lines
    .filter((line) => /^user:\s*/i.test(line))
    .map((line) => line.replace(/^user:\s*/i, "").trim())
    .filter((line) => line.length > 5);

  const summary = userLines.slice(-6).join("; ").slice(0, 500) || lines.slice(-6).join("; ").slice(0, 500);

  // Extract facts: statements containing possessive/ownership/identity patterns
  const facts = userLines
    .filter((line) => /我[的有是在]|我叫|我用|my |i have|i am|i use|我們|名字/i.test(line))
    .map((line) => `用戶: ${line.slice(0, 200)}`)
    .slice(0, 5);

  // Extract preferences: likes, preferences, habits
  const preferences = userLines
    .filter((line) => /喜歡|偏好|習慣|prefer|like|love|favorite|always use|都用/i.test(line))
    .map((line) => `用戶偏好: ${line.slice(0, 200)}`)
    .slice(0, 5);

  // Extract decisions
  const decisions = userLines
    .filter((line) => /決定|改用|選擇|will use|going to|打算|要用|換成/i.test(line))
    .map((line) => `用戶決定: ${line.slice(0, 200)}`)
    .slice(0, 3);

  // Extract environment: hardware, software, system mentions
  const envTruths = userLines
    .filter((line) => /跑|裝|安裝|deploy|install|server|GPU|CPU|docker|k8s|port|系統|主機/i.test(line))
    .map((line) => `用戶環境: ${line.slice(0, 200)}`)
    .slice(0, 3);

  // Extract pitfalls
  const pitfalls = userLines
    .filter((line) => /不要|避免|坑|error|fail|bug|壞|問題|crash/i.test(line))
    .map((line) => `問題: ${line.slice(0, 200)}`)
    .slice(0, 3);

  // Extract todos
  const todos = userLines
    .filter((line) => /待辦|todo|接下來|next|需要|之後|later|還沒/i.test(line))
    .map((line) => line.slice(0, 200))
    .slice(0, 3);

  // Extract corrections: user says "不對/wrong/不是" then gives correct answer
  const corrections = userLines
    .filter((line) => /不對|不是|wrong|actually|其實|應該是|correct/i.test(line))
    .map((line) => `糾正: ${line.slice(0, 200)}`)
    .slice(0, 3);

  // Extract best practices: approaches that worked
  const bestPractices = userLines
    .filter((line) => /這樣做|work|有效|成功|解決|搞定|OK了|可以了|這招/i.test(line))
    .map((line) => `有效做法: ${line.slice(0, 200)}`)
    .slice(0, 3);

  return normalizeOutput({
    ...DEFAULT_OUTPUT,
    session_summary: summary,
    confirmed_facts: facts,
    decisions,
    pitfalls,
    preference_updates: preferences,
    environment_truths: envTruths,
    open_loops: todos,
    corrections,
    best_practices: bestPractices,
    scope_recommendation: "owner_shared",
  });
}

const DISTILL_SYSTEM_PROMPT = `You are a memory distillation engine that extracts ONLY user-stated facts, preferences, and decisions from a conversation transcript. Return valid JSON only — no markdown, no commentary, no \`\`\` fencing.

## Critical Rules

1. Extract ONLY information the USER stated or confirmed — never extract assistant suggestions, questions, or filler.
2. Write each memory as a clean, standalone, third-person statement about the user. Examples:
   - Good: "用戶喜歡 C#" / "User prefers C#"
   - Good: "用戶的貓叫米露" / "User's cat is named 米露"
   - Bad: "I like C#" (raw transcript fragment)
   - Bad: "The assistant suggested using TypeScript" (assistant action, not user fact)
3. Preserve the user's original language. If the user speaks Chinese, write memories in Chinese. If mixed, prefer the user's primary language.
4. Be specific and factual. Include names, versions, quantities when the user mentions them.
5. DO NOT extract:
   - Greetings, small talk, or conversational filler
   - Questions the user asked (unless the question reveals a fact, e.g. "我的 J1900 能跑這個嗎？" → "用戶擁有 J1900 嵌入式系統")
   - Assistant responses or recommendations
   - Temporary debugging steps or one-time commands

## JSON Schema

{
  "session_summary": "1-2 sentence summary of what the user discussed/accomplished",
  "confirmed_facts": ["standalone facts about the user, their systems, pets, possessions, work, etc."],
  "decisions": ["decisions the user made during this session"],
  "pitfalls": ["problems or bugs the user encountered"],
  "preference_updates": ["user preferences, likes, dislikes, habits, routines"],
  "environment_truths": ["user's hardware, software, infrastructure, accounts, services"],
  "open_loops": ["tasks the user mentioned but didn't complete"],
  "corrections": ["things the user corrected about the assistant's response or approach"],
  "best_practices": ["approaches or patterns that worked well and should be reused"],
  "scope_recommendation": "owner_shared"
}

## Field Guidelines

- confirmed_facts: Personal facts — name, pets, family, possessions, hobbies, professional background
- preference_updates: Likes/dislikes, preferred tools, payment methods, habits (e.g. "用戶在麥當勞用點點卡付款")
- environment_truths: Hardware, OS, servers, accounts, installed software, API keys, services
- decisions: Choices made — "decided to use X", "chose Y over Z", "will deploy with W"
- open_loops: Unfinished tasks, things to try later, follow-ups mentioned
- pitfalls: Bugs hit, errors encountered, things that didn't work
- corrections: When the user says "不對/wrong/不是那樣" and provides the right answer. Write as: "正確做法：X（而非 Y）"
- best_practices: Approaches that solved a problem well. Write as: "做 X 時應該用 Y 方法" — only extract if the user confirmed it works
- scope_recommendation: Almost always "owner_shared" — use "agent_local" only for agent-specific config`;

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
      max_tokens: 4096,
      messages: [
        {
          role: "system",
          content: DISTILL_SYSTEM_PROMPT,
        },
        {
          role: "user",
          content:
            `${customInstructions ? `${customInstructions}\n\n` : ""}Distill the following conversation into structured memory JSON. Extract ONLY what the USER said or confirmed:\n\n${transcript}`,
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
    model: config.model ?? process.env.OPENCLAW_DISTILL_MODEL ?? "gpt-4o-mini",
    baseURL: config.baseURL ?? process.env.OPENCLAW_DISTILL_BASE_URL ?? "https://api.openai.com/v1",
    apiKey: config.apiKey ?? process.env.OPENCLAW_DISTILL_API_KEY ?? "",
    ...config,
  };

  return {
    async distillTranscript(transcript: string, opts?: { customInstructions?: string }): Promise<DistillOutput> {
      if (resolved.distillImpl) {
        return normalizeOutput(await resolved.distillImpl(transcript, opts));
      }

      try {
        const raw = await callChatCompletions(resolved, transcript, opts?.customInstructions);
        const parsed = parseDistillJson(raw);
        if (parsed) {
          console.log(`[memory-lancedb-brain] distill: LLM OK — facts=${parsed.confirmed_facts.length} prefs=${parsed.preference_updates.length} env=${parsed.environment_truths.length} decisions=${parsed.decisions.length}`);
          return parsed;
        }
        console.warn(`[memory-lancedb-brain] distill: LLM returned unparseable JSON, falling back to heuristic. Raw (first 200): ${raw.slice(0, 200)}`);
        return heuristicDistill(transcript);
      } catch (err) {
        console.warn(`[memory-lancedb-brain] distill: LLM call failed, falling back to heuristic: ${err instanceof Error ? err.message : String(err)}`);
        return heuristicDistill(transcript);
      }
    },
  };
}
