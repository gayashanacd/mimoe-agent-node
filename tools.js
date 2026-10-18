// Tools the agent can call. Each tool is a plain JavaScript function plus
// an OpenAI-format JSON schema so the model knows how to call it.

import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const NOTES_FILE = path.join(__dirname, "notes.txt");

// ---------- calculator ----------
// A tiny arithmetic parser. We never use eval() on text the model produced,
// because the model could output code instead of math.
export function calculator({ expression }) {
  try {
    const tokens = String(expression).match(/\d+\.?\d*|[-+*/%^()]/g);
    if (!tokens || tokens.join("") !== String(expression).replace(/\s+/g, "")) {
      return "error: unsupported expression";
    }
    let pos = 0;
    const peek = () => tokens[pos];
    const next = () => tokens[pos++];

    // grammar: expr = term (('+'|'-') term)*
    //          term = power (('*'|'/'|'%') power)*
    //          power = unary ('^' power)?
    //          unary = '-' unary | primary
    //          primary = number | '(' expr ')'
    const expr = () => {
      let v = term();
      while (peek() === "+" || peek() === "-") v = next() === "+" ? v + term() : v - term();
      return v;
    };
    const term = () => {
      let v = power();
      while (["*", "/", "%"].includes(peek())) {
        const op = next();
        const r = power();
        v = op === "*" ? v * r : op === "/" ? v / r : v % r;
      }
      return v;
    };
    const power = () => {
      const base = unary();
      return peek() === "^" ? (next(), base ** power()) : base;
    };
    const unary = () => (peek() === "-" ? (next(), -unary()) : primary());
    const primary = () => {
      const t = next();
      if (t === "(") {
        const v = expr();
        if (next() !== ")") throw new Error("missing )");
        return v;
      }
      const n = Number(t);
      if (Number.isNaN(n)) throw new Error(`unexpected '${t}'`);
      return n;
    };

    const result = expr();
    if (pos !== tokens.length) throw new Error("unexpected trailing input");
    return String(result);
  } catch (e) {
    return `error: ${e.message}`;
  }
}

// ---------- current time ----------
export function getCurrentTime() {
  return new Date().toLocaleString("en-US", { dateStyle: "full", timeStyle: "short" });
}

// ---------- local notes search ----------
// A tiny on-device "retrieval": keyword match over notes.txt.
// The data never leaves the machine.
export function searchNotes({ query }) {
  if (!existsSync(NOTES_FILE)) return "No notes file found.";
  const words = String(query).toLowerCase().split(/\s+/).filter((w) => w.length > 2);
  const hits = readFileSync(NOTES_FILE, "utf8")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && words.some((w) => l.toLowerCase().includes(w)));
  return hits.length ? hits.slice(0, 5).join("\n") : "No matching notes.";
}

const TOOLS = {
  calculator,
  get_current_time: getCurrentTime,
  search_notes: searchNotes,
};

export function runTool(name, args = {}) {
  const fn = TOOLS[name];
  if (!fn) return `error: unknown tool '${name}'`;
  try {
    return fn(args);
  } catch (e) {
    return `error: ${e.message}`;
  }
}

// ---------- OpenAI-format schemas ----------
export const TOOL_SCHEMAS = [
  {
    type: "function",
    function: {
      name: "calculator",
      description: "Evaluate an arithmetic expression, e.g. '23 * (4 + 5)'.",
      parameters: {
        type: "object",
        properties: { expression: { type: "string" } },
        required: ["expression"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_current_time",
      description: "Get the current local date and time (now). Not for looking up the user's events.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "search_notes",
      description:
        "Search the user's personal notes. Use for ANY question about the user's own " +
        "appointments, schedule, meetings, deadlines or other personal facts.",
      parameters: {
        type: "object",
        properties: { query: { type: "string" } },
        required: ["query"],
      },
    },
  },
];
