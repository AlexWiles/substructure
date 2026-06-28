# plan-mode

A two-phase agent: a planning phase where the user and the model
iterate on a TODO list over many turns, then an execution phase where
the model walks the list end-to-end. Inspired by the "plan mode"
interaction in coding agents like Claude Code.

## What it shows

- **`client.action` payloads** as a non-message way for the client to
  drive state changes. Sending `{ type: "action", name: "set_mode", ... }`
  flips state inside the worker without producing a user message in
  the transcript.
- **Selector-form middleware** (`agent.tools`, `agent.llm`) reading
  the same state slice (`mode`) so the toolset and model swap together
  when the mode changes.
- **A custom history middleware** built from the `triggerToMessage` and
  `prependHistoryToLlmCalls` SDK helpers. This one resets the
  transcript whenever it observes a mode transition and computes its
  own mode-dependent system prompt, so the executor starts cold with
  only the rendered plan in its system prompt.
- **Embedded runtime** with session persistence: subsequent CLI
  invocations resume the same conversation by passing the session id.

## Architecture

One shared state shape:

```ts
type PlanState = {
    mode: "planning" | "executing";
    plan: { goal: string; steps: Step[]; nextId: number };
    messages: Message[];
    lastMode?: "planning" | "executing";
};
```

Two middlewares:

- **`planMode`** — on a `set_mode` client action, writes the new mode
  to state and lets the chain proceed normally. The action itself
  becomes the trigger that kicks off execution.
- **`modeAwareHistory`** — records triggers to `state.messages`, computes
  a mode-dependent system prompt, and prepends both (system message
  first, then the transcript) to outgoing `call.llm` actions. If
  `state.mode !== state.lastMode`, wipes `messages` first.

Chain order:

```
planMode → modeAwareHistory → tools → llm
```

`modeAwareHistory` puts its system prompt ahead of the transcript so it
ends up first in the LLM request. Tool gating and model selection use
the selector forms — planning runs on Opus 4.7 (harder reasoning),
executing runs on Sonnet 4.6 (mechanical step-by-step work).

The domain is intentionally generic. Planning tools edit a TODO list
(`set_goal`, `add_step`, `update_step`, `remove_step`). Executing
mode exposes one tool — `complete_step(id, note)` — which marks a
step done and records a short note about how it was handled. Replace
that with whatever real work your agent actually does.

## Run

```sh
export OPENROUTER_API_KEY=sk-or-...
pnpm install

# Generate a session id, then reuse it across calls.
SESSION=$(uuidgen)

pnpm tsx index.ts $SESSION "Plan a weekend trip to Lisbon."
pnpm tsx index.ts $SESSION "Add a step for booking a city tour."
pnpm tsx index.ts $SESSION "Combine the food steps into one."

# Flip mode. This action also kicks off execution; the model walks the list.
pnpm tsx index.ts $SESSION "/mode executing"
```

Each invocation streams events to stdout as they happen: LLM call
requests, LLM responses, tool calls, tool results, and a
`turn.completed` line with cost and token usage. The session lives in
`agent.db` (a local SQLite file) so state persists across runs.

## Adapt

The middlewares only care about `mode` and `plan`. To repurpose:

- Replace `planningTools` / `executingTools` with your domain tools.
- Replace the planning / execution system prompts.
- Add more modes (e.g. `reviewing`, `debugging`) by extending the
  `Mode` union and registering another toolset and prompt branch.
- Swap the model selection in `agent.llm` for your own routing.
