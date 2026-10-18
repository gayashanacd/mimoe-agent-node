// Offline tests — no mimOE needed. Run with: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { calculator, searchNotes, runTool } from "../tools.js";
import { parseTextToolCall, stripThinking, trimHistory, skippedTools } from "../agent.js";

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
  assert.match(searchNotes({ query: "zzzz" }), /^No exact match\. All notes:\n[\s\S]*standup/i);
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

test("trimHistory drops whole oldest turns, keeps system + current turn", () => {
  const sys = { role: "system", content: "s" };
  const old = [
    { role: "user", content: "x".repeat(100) },
    { role: "assistant", content: "", tool_calls: [{ id: "1" }] },
    { role: "tool", tool_call_id: "1", content: "y".repeat(100) },
    { role: "assistant", content: "done" },
  ];
  const current = { role: "user", content: "now" };
  const h = trimHistory([sys, ...old, current], 120);
  assert.deepEqual(h, [sys, current]);
  // never removes the current turn, even if it alone is too big
  const big = { role: "user", content: "z".repeat(500) };
  assert.deepEqual(trimHistory([sys, big], 10), [sys, big]);
});

test("skippedTools spots an un-called tool or a question back to the user", () => {
  assert.ok(skippedTools("I should check your notes to see what you have scheduled."));
  assert.ok(skippedTools("Let me search for it."));
  assert.ok(skippedTools("If you have any notes or appointments, please provide them."));
  assert.ok(!skippedTools("Your dentist appointment is on October 28 at 3 PM."));
});
