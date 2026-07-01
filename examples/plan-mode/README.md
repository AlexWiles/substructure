# plan-mode

A two-phase agent: a planning phase where the user and the model
iterate on a TODO list over many turns, then an execution phase where
the model walks the list end-to-end. Inspired by the "plan mode"
interaction in coding agents like Claude Code.

## What it shows

- **`client.action` payloads** as a non-message way for the client to
  drive state changes. Sending `{ type: "action", name: "set_mode", ... }`
  flips the mode inside the worker without producing a user message in
  the transcript.
- **A custom `decide` function as a modal agent**: one decision function reads its
  current `mode` from state and picks the model, system prompt, and
  toolset for that mode, so they swap together when the mode changes.
- **Forking a fresh branch on a mode transition**: entering `executing`
  seeds a brand-new transcript — fresh message ids branch the thread —
  holding only the rendered plan, so the executor starts cold instead of
  inheriting the planning back-and-forth.
- **Embedded runtime** with session persistence: subsequent CLI
  invocations resume the same conversation by passing the session id.

## Architecture

One shared state shape, which rides the wire as `worker_state` (the engine
owns the conversation history — it isn't part of state):

```ts
type State = {
    mode: "planning" | "executing";
    plan: { goal: string; steps: Step[]; nextId: number };
};
```

The agent is a single custom `decide` function — `agent({ name: "planner",
decide })`. There are no middlewares. Each decision defaults `mode`/`plan`
from `req.state`, looks up a per-mode profile, and builds the SDK's default
`toolLoop` for that mode.

A `profiles` table maps each mode to its model and system prompt:

- **planning** runs on Opus 4.7 (harder reasoning) with the plan-editing
  tools (`set_goal`, `add_step`, `update_step`, `remove_step`).
- **executing** runs on Sonnet 4.6 (mechanical step-by-step work) with
  one tool, `complete_step`.

`planTools(state)` builds the toolset for the current mode, closing over the
live `state` so each `execute` mutates `state.plan` directly. The chosen
model, prompt, and tools go into a `toolLoop`, which owns the actual
conversation — prompting, the LLM call, the tool round-trip, and round
gating. The `decide` runs that loop for everything except one case: the
`set_mode` action.

On `set_mode` (a `client.action`), the `decide` writes the requested mode to
state. If it's *entering* `executing`, it forks a fresh thread: it replaces
`req.transcript` with an empty one and synthesizes a `user.message` trigger
holding only `renderPlan(plan)`, so the executing loop roots a brand-new
thread with its own system prompt and sees only the rendered plan, none of
the planning chatter. Otherwise the loop just re-prompts under the new mode
against existing history. Either way the same loop runs at the end, so
switching to `executing` immediately starts the model walking the list.

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

The `decide` only cares about `mode` and `plan`. To repurpose:

- Replace the `planning` / `executing` tool lists in `planTools` with your domain tools.
- Replace the planning / execution system prompts.
- Add more modes (e.g. `reviewing`, `debugging`) by extending the
  `Mode` union and adding another entry to the `profiles` table.
- Swap the per-mode `model` in `profiles` for your own routing.
