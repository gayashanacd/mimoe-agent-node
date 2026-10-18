// A small tool-using AI agent that runs entirely on-device against
// mimOE's local, OpenAI-compatible inference endpoint.
//
// Approach: "BYO Framework" with raw API calls via the official OpenAI
// Node SDK. The only thing that changes vs. a cloud model is baseURL.
//
// Run:  node agent.js              (interactive chat)
//       node agent.js "question"   (one-shot)

import "dotenv/config";
import OpenAI from "openai";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { TOOL_SCHEMAS, runTool } from "./tools.js";

const BASE_URL = process.env.MIMOE_BASE_URL ?? "http://localhost:8083/mimik-ai/openai/v1";
const API_KEY = process.env.MIMOE_API_KEY ?? "not-needed"; // key from Studio's API panel
const MODEL = process.env.MIMOE_MODEL ?? "smollm2-360m";   // exact model id from Studio
const MAX_STEPS = Number(process.env.MAX_STEPS ?? 5);      // guard against infinite tool loops
const DEBUG = process.env.DEBUG === "1";
const THINKING = process.env.THINKING === "1";             // Qwen3 reasoning mode (slower, uses more context)
const MAX_TOKENS = Number(process.env.MAX_TOKENS ?? 512);  // cap each reply so it can't overflow the context
// ~3.5 chars per token: 4000 chars of history + tool schemas + a 512-token reply fit in a 2048-token context.
const MAX_HISTORY_CHARS = Number(process.env.MAX_HISTORY_CHARS ?? 4000);

const client = new OpenAI({ baseURL: BASE_URL, apiKey: API_KEY });

const TODAY = new Date().toLocaleDateString("en-US", { dateStyle: "full" });

export const SYSTEM_PROMPT = `You are a helpful assistant running locally on the user's device.
Today is ${TODAY}.
You can use these tools: calculator, get_current_time, search_notes.
Use a tool when the question needs exact math, the current time, or the user's notes.
The user keeps notes about their appointments, meetings, team routines and deadlines.
If a question could be answered by those notes, call search_notes before answering; never say the information is unavailable without searching first.

If your runtime supports function calling, call the tool normally.
Otherwise, to use a tool reply with ONLY a single line of JSON, for example:
{"tool": "calculator", "arguments": {"expression": "12 * 7"}}
When you have the final answer, reply in plain text (no JSON).${THINKING ? "" : "\n/no_think"}`;

const log = (...args) => DEBUG && console.error("[debug]", ...args);

// Reasoning models like Qwen3 wrap their thinking in <think>...</think>.
// Strip it so the user sees only the answer and it doesn't fill the small
// on-device context window (2048 tokens on a 2 GB GPU) on later turns.
export function stripThinking(content) {
  return (content ?? "").replace(/<think>[\s\S]*?(<\/think>|$)/g, "").trim();
}

// Fallback for small models without native function calling:
// look for a {"tool": ..., "arguments": ...} JSON object in the text.
export function parseTextToolCall(content) {
  const match = (content ?? "").match(/\{[\s\S]*\}/);
  if (!match) return null;
  try {
    const data = JSON.parse(match[0]);
    if (data && typeof data === "object" && "tool" in data) {
      return { name: data.tool, args: data.arguments ?? {} };
    }
  } catch {
    /* not JSON — treat as a normal answer */
  }
  return null;
}

// Small models sometimes skip their tools: they *say* they'll use one
// ("Let me check your notes") without calling it, or ask the user for
// information the notes already hold. Spot that so the loop can nudge them once.
export function skippedTools(content) {
  const text = content ?? "";
  const announces =
    /\b(let me|i'll|i will|i should|i need to|i can)\b.{0,40}\b(check|search|look|calculat|find)/i;
  const asksUser =
    /(please provide|could you (please )?(share|provide|tell)|i need to know|not (provided|available|specified|recorded|mentioned)|i don't have (any )?information)/i;
  return announces.test(text) || asksUser.test(text);
}

// Keep the conversation inside the small context window: drop whole oldest
// turns (each starts at a user message), always keeping the system prompt and
// the current turn. Turns are dropped whole so a tool result is never
// separated from the assistant message that requested it.
export function trimHistory(history, maxChars = MAX_HISTORY_CHARS) {
  const size = () => history.reduce((n, m) => n + JSON.stringify(m).length, 0);
  while (size() > maxChars) {
    const nextTurn = history.findIndex((m, i) => i > 1 && m.role === "user");
    if (nextTurn === -1) break; // only the current turn left
    history.splice(1, nextTurn - 1);
  }
  return history;
}

// One agent turn: think -> (maybe) act with a tool -> observe -> answer.
export async function runAgent(userInput, history) {
  history.push({ role: "user", content: userInput });
  trimHistory(history);
  let usedTool = false;
  let nudged = false;

  for (let step = 0; step < MAX_STEPS; step++) {
    const response = await client.chat.completions.create({
      model: MODEL,
      messages: history,
      tools: TOOL_SCHEMAS,
      temperature: 0.2, // low temperature = more reliable tool use from small models
      max_tokens: MAX_TOKENS,
    });
    const msg = response.choices[0].message;
    log(`step ${step}:`, JSON.stringify(msg));
    msg.content = stripThinking(msg.content);

    // Path 1: native OpenAI-style tool calls
    if (msg.tool_calls?.length) {
      history.push(msg);
      for (const call of msg.tool_calls) {
        let args;
        try {
          args = JSON.parse(call.function.arguments || "{}");
        } catch {
          args = null;
        }
        const result = args
          ? runTool(call.function.name, args)
          : `error: arguments were not valid JSON: ${call.function.arguments}`;
        console.log(`  🔧 ${call.function.name}(${JSON.stringify(args)}) -> ${result}`);
        history.push({ role: "tool", tool_call_id: call.id, content: result });
      }
      usedTool = true;
      continue;
    }

    // Path 2: text-protocol tool call (fallback for small models)
    const parsed = parseTextToolCall(msg.content);
    if (parsed) {
      const result = runTool(parsed.name, parsed.args);
      console.log(`  🔧 ${parsed.name}(${JSON.stringify(parsed.args)}) -> ${result}`);
      history.push({ role: "assistant", content: msg.content });
      history.push({
        role: "user",
        content: `Tool result for ${parsed.name}: ${result}\nNow answer the original question in plain text.`,
      });
      usedTool = true;
      continue;
    }

    // Path 3: the model skipped its tools (see skippedTools). Nudge once.
    if (!usedTool && !nudged && skippedTools(msg.content)) {
      log("model answered without using its tools; nudging");
      nudged = true;
      history.push({ role: "assistant", content: msg.content });
      history.push({
        role: "user",
        content: "Don't ask me. Use your tools (e.g. search_notes) to find out, then answer.",
      });
      continue;
    }

    // Path 4: final answer
    const answer = msg.content;
    history.push({ role: "assistant", content: answer });
    return answer;
  }

  return "Stopped: reached the maximum number of reasoning steps.";
}

// Turn SDK errors into a hint about what to check in mimOE Studio.
function explainError(e) {
  if (e instanceof OpenAI.APIConnectionError) {
    return `can't reach ${BASE_URL}. Is mimOE Studio running with a model loaded?`;
  }
  if (e.status === 401) return "unauthorized. Check MIMOE_API_KEY against Studio's API panel.";
  if (e.status === 404) return `not found. Is '${MODEL}' loaded? Check MIMOE_MODEL against Studio.`;
  return e.message;
}

async function main() {
  const history = [{ role: "system", content: SYSTEM_PROMPT }];
  console.log(`mimOE local agent  |  model=${MODEL}  |  endpoint=${BASE_URL}`);

  const question = process.argv.slice(2).join(" ");
  if (question) {
    console.log(await runAgent(question, history));
    return;
  }

  const rl = readline.createInterface({ input, output });
  console.log("Type a question (or 'exit').");
  while (true) {
    let userInput;
    try {
      userInput = (await rl.question("\nyou> ")).trim();
    } catch {
      break; // input closed (Ctrl+D / Ctrl+Z / end of piped input)
    }
    if (["exit", "quit"].includes(userInput.toLowerCase())) break;
    if (!userInput) continue;
    const turnStart = history.length;
    try {
      console.log("agent>", await runAgent(userInput, history));
    } catch (e) {
      history.length = turnStart; // drop the failed turn so a retry starts clean
      console.error("error talking to mimOE:", explainError(e));
    }
  }
  rl.close();
}

// Only start the chat when run directly (not when imported by tests).
if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("agent.js")) {
  main().catch((e) => {
    console.error("Fatal:", explainError(e));
    process.exit(1);
  });
}
