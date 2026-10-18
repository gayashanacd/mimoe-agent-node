// Offline tests — no mimOE needed. Run with: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { calculator, searchNotes, runTool } from "../tools.js";
import { parseTextToolCall, stripThinking } from "../agent.js";

test("calculator does arithmetic", () => {
  assert.equal(calculator({ expression: "1234 * 56" }), "69104");
  assert.equal(calculator({ expression: "2 + 3 * (4 - 1)" }), "11");
  assert.equal(calculator({ expression: "-2 ^ 2" }), "4");
});

test("calculator refuses code", () => {
  assert.match(calculator({ expression: "process.exit()" }), /^error/);
});

test("searchNotes finds matching lines", () => {
  assert.match(searchNotes({ query: "dentist" }), /October 28/);
  assert.equal(searchNotes({ query: "zzzz" }), "No matching notes.");
});

test("unknown tool returns an error", () => {
  assert.match(runTool("nope"), /unknown tool/);
});

test("fallback parser reads JSON tool calls from text", () => {
  assert.deepEqual(
    parseTextToolCall('Sure! {"tool": "calculator", "arguments": {"expression": "2+2"}}'),
    { name: "calculator", args: { expression: "2+2" } }
  );
  assert.equal(parseTextToolCall("The answer is 4."), null);
});

test("stripThinking removes Qwen3 <think> blocks", () => {
  assert.equal(stripThinking("<think>\nhmm...\n</think>\n\nThe answer is 42."), "The answer is 42.");
  assert.equal(stripThinking("<think>cut off mid-thought"), "");
  assert.equal(stripThinking(null), "");
});
