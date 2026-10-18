# mimOE Local Agent (Node.js)

A small tool-using AI agent that runs **entirely on-device**, using the
OpenAI-compatible inference endpoint exposed by **mimOE Studio**.

## What it does
A chat agent that decides on its own when to use three tools:

| Tool | Why it's there |
|---|---|
| `calculator` | Small LLMs are bad at exact math — the agent offloads it |
| `get_current_time` | The model has no clock — shows grounding in live data |
| `search_notes` | Keyword search over a local `notes.txt` — private data that never leaves the device |

Example:
```
you> What is 1234 * 56, and when is my dentist appointment?
  🔧 calculator({"expression":"1234 * 56"}) -> 69104
  🔧 search_notes({"query":"dentist appointment"}) -> Dentist appointment on October 28 at 3 PM.
agent> 1234 × 56 = 69,104. Your dentist appointment is October 28 at 3 PM.
```

## Architecture
```
 you ──► agent.js (agent loop) ──HTTP──► mimOE Studio  /mimik-ai/openai/v1/chat/completions
              │                              (Qwen3-1.7B running locally)
              └──► tools.js (calculator, time, local notes)
```
The loop: send history + tool schemas → if the model requests a tool, run it,
append the result, and ask again → stop when the model gives a plain answer
(or after `MAX_STEPS`).

## Setup
Requires Node.js 18+.

1. Install and open **mimOE Studio**. Go to **AI Models -> On Device** and download
   **qwen3-1.7b** (listed as "tool use support").
2. Load it. On a small GPU, set the **context size to 4096** first (see "What I observed").
3. Click the **API** button and note the base URL, API key, and model id.
3. Then:
```bash
npm install
cp .env.example .env        # fill in values from step 3 (Windows: copy .env.example .env)
npm start                   # interactive chat
node agent.js "What is 17% of 2450?"   # one-shot
npm test                    # offline tests for the tools (no mimOE needed)
```
Set `DEBUG=1` in `.env` to see raw model responses, and `THINKING=1` to let Qwen3 reason before answering.

## Design choices
- **Raw API calls via the official OpenAI Node SDK, no framework.** The task is
  small; a ~100-line loop is easier to read, debug, and explain than LangChain
  abstractions. Because mimOE speaks the OpenAI protocol, switching from a
  cloud model to local is just the `baseURL` — that's the core point.
- **Two tool-calling paths.** Native OpenAI `tool_calls` first; if the small
  model doesn't emit them, a fallback parses a JSON `{"tool": ..., "arguments": ...}`
  object from plain text. Small on-device models are often unreliable at function calling.
- **Safety.** The calculator uses a tiny hand-written parser instead of `eval()` —
  never execute model output. `MAX_STEPS` prevents infinite tool loops.
- **Low temperature (0.2)** for more consistent tool selection.
- **Strip `<think>` blocks.** Qwen3 is a reasoning model. Its thinking is removed from
  the answer and from the chat history, so it doesn't fill the 4096-token context on
  later turns. `THINKING=0` (default) appends Qwen3's `/no_think` switch for speed.
- **Tool descriptions are part of the prompt.** With a 1.7B model, wording mattered more
  than code: changing `search_notes` to "use for ANY question about the user's own
  appointments, schedule..." fixed a case where the model answered from the clock instead.
- **Config via `.env`**, never committed.

## What I observed / limitations
Tested on a Windows 11 laptop: 16 GB RAM, **NVIDIA MX150 with 2 GB VRAM**.

- **No model was bundled in my Studio build.** AI Models was empty; the On Device tab
  offered qwen3-1.7b, qwen3-4b and smollm-360m to download. I picked **qwen3-1.7b**: it's
  labelled tool-use capable and small enough for my GPU. SmolLM-360M is usually too small
  for reliable function calling.
- **Load failed: `failed to create the llama_context`.** Studio's default context was 12000
  tokens. The KV cache for that (~1.3 GB) plus the weights (~1.1 GB) doesn't fit in 2 GB of
  VRAM. Lowering context to **4096** fixed it. `GET /mimik-ai/openai/v1/models` confirms
  `max_context: 4096`, all layers on GPU, ~28 tokens/s.
- **The endpoint really is OpenAI-compatible.** The official `openai` npm SDK worked
  unchanged, including **native `tool_calls`**. The text-JSON fallback was never needed
  with Qwen3; it's kept for smaller models.
- **Multi-tool calls work.** For "What is 1234 * 56, and when is my dentist appointment?"
  the model called `calculator` and `search_notes` in one step and answered correctly.
- **Thinking vs. no thinking** (same 4 questions):

  | Mode | Typical latency | Tool choice |
  |---|---|---|
  | `THINKING=1` | 10–30 s | Good |
  | `THINKING=0` (`/no_think`) | 5–13 s | Good after improving the tool descriptions |

- **Small-model limits.** Tool use isn't fully reliable. "What time is the team standup?"
  sometimes searched the notes correctly. Other times the model said "not available", or
  said "Let me search for it" without actually calling the tool. Follow-up questions in a
  long chat were more likely to skip tools. Answers sometimes contain LaTeX (`$69104$`).
- **The GPU driver crashed under load.** Partway through a later question, the mimOE engine
  (`mimoe.exe`) aborted. Afterwards even `nvidia-smi` segfaulted. On restart, the engine's
  Vulkan backend fell back to the Intel iGPU and crashed while loading tensors (found via
  `~/.mimoe/.edge/logs/mimoe.log` and the Windows event log). Rebooting reset the driver. The
  agent now reports "can't reach mimOE" instead of a bare `Connection error`.
- **The API key is required.** Requests without it return `401 Unauthorized`.

## Next steps
- Try mimik's own `@mimik/agent-kit` to compare with the BYO approach.
- Retry once when the model says it will use a tool but doesn't call one.
- Try qwen3-4b on a machine with more VRAM and compare tool reliability.
- Use an embeddings endpoint (if available) for semantic note search.
- Stream responses; add a simple web UI.
