---
title: Patterns
---

Things you can build out of tools, custom decision functions, and client actions. Each links to a runnable example.

## System prompt

Instructions are worker config, not conversation. Keep the tree pure conversation: never store a system message in it. Prepend the instructions to each `llm.call`'s `messages` at build time — the exact prompt sent is durably recorded on the `llm.call.requested` event, so nothing is lost. Record the instructions (or their version) in `state.system` so their history is branch-aware: because state is [branch-scoped](./02-concepts.md#state-is-branch-scoped), changing `state.system` is "branching at the system prompt" — same conversation, new instructions, and the old branch still resolves to the version that governed it.

## Memory

Keep agent memory (facts learned, rolling notes) in `state.memory` and fold it into the prompt at build time. Because state versions anchor to the tree, a fork back to an earlier message automatically resolves memory as of that point: the new branch is uncontaminated by anything learned on the abandoned one, with no bookkeeping in your worker.

## Compaction

To bound the prompt without losing history, keep a summary in state with a cutoff: `state.summary = { text, through: node_id }`. Build each prompt as `[system, summary] ++ messages after through`; the tree keeps the full conversation. Since the summary lives in state, a fork before `through` resolves to an earlier summary version (or none), so a summary never describes messages that aren't on the branch.

## Cross-session data

A session's state lives only for that session. To persist data across sessions (a user's todo list, customer preferences, a counter), store it in your own database keyed by user id. Tools are pure functions that reach the store directly through the decision request, keyed by `request.identity.id`, so the data follows the user across every session, and never rides the wire.

Example: [`examples/typescript-sdk-hybrid-state`](https://github.com/substructureai/substructure/tree/main/examples/typescript-sdk-hybrid-state). Walkthrough: [SDK / State](./04-sdk.md#state).

## Tool approval (human-in-the-loop)

Pause the agent before a sensitive tool call (one that spends money, mutates production, shells out, sends a message). This is a custom `decide`: when the model calls the tool, `decide` parks the request in its state and ends the turn instead of running it. The client UI shows the pending request. The user approves or denies with a `client.action` (`decide` switches on `req.trigger.type === "client.action"`). On approval `decide` re-emits the original call with the same `tool_call_id`, so the LLM sees one tool call and one result.

Example: [`examples/typescript-sdk-tool-approval`](https://github.com/substructureai/substructure/tree/main/examples/typescript-sdk-tool-approval).

## Modal agents (plan mode)

Run the agent in distinct phases: a planner that iterates on a TODO list with the user, then an executor that walks the list end-to-end. The agent is a custom `decide` that reads a `mode` field from state and picks the prompt, tools, and model for that mode. The client flips the mode with a `client.action`, not a chat message, so the switch doesn't appear in the transcript. Fork a fresh branch at the boundary (seeded with just the plan) if you want the executor to start clean.

Example: [`examples/typescript-sdk-plan-mode`](https://github.com/substructureai/substructure/tree/main/examples/typescript-sdk-plan-mode).
