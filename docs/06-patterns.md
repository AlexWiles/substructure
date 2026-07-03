---
title: Patterns
---

Things you can build out of tools, custom decision functions, and client actions. Each links to a runnable example.

## Cross-session data

A session's state lives only for that session. To persist data across sessions (a user's todo list, customer preferences, a counter), store it in your own database keyed by user id. Tools are pure functions that reach the store directly through the decision request — keyed by `request.identity.id` — so the data follows the user across every session, and never rides the wire.

Example: [`examples/hybrid-state`](https://github.com/substructureai/substructure/tree/main/examples/hybrid-state). Walkthrough: [SDK / State](./04-sdk.md#state).

## Tool approval (human-in-the-loop)

Pause the agent before a sensitive tool call (one that spends money, mutates production, shells out, sends a message). This is a custom `decide`: when the model calls the tool, `decide` parks the request in its state and ends the turn instead of running it. The client UI shows the pending request. The user approves or denies with a `client.action` (`decide` switches on `req.trigger.type === "client.action"`). On approval `decide` re-emits the original call with the same `tool_call_id`, so the LLM sees one tool call and one result.

Example: [`examples/tool-approval`](https://github.com/substructureai/substructure/tree/main/examples/tool-approval).

## Modal agents (plan mode)

Run the agent in distinct phases: a planner that iterates on a TODO list with the user, then an executor that walks the list end-to-end. The agent is a custom `decide` that reads a `mode` field from state and picks the prompt, tools, and model for that mode. The client flips the mode with a `client.action`, not a chat message, so the switch doesn't appear in the transcript. Fork a fresh branch at the boundary (seeded with just the plan) if you want the executor to start clean.

Example: [`examples/plan-mode`](https://github.com/substructureai/substructure/tree/main/examples/plan-mode).
