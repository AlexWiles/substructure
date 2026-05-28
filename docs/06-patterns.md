---
title: Patterns
---

Things you can build out of middleware, tools, and client actions. Each links to a runnable example.

## Cross-session data

A session's state lives only for that session. To persist data across sessions (a user's todo list, customer preferences, a counter), store it in your own database keyed by user id. A middleware loads it at the start of every decision and saves at the end. Tools see normal in-memory state and don't know about the round-trip.

Example: [`examples/hybrid-state`](https://github.com/substructureai/substructure/tree/main/examples/hybrid-state). Walkthrough: [SDK / Hybrid wire and database state](./04-sdk.md#example-hybrid-wire-and-database-state).

## Tool approval (human-in-the-loop)

Pause the agent before a sensitive tool call (one that spends money, mutates production, shells out, sends a message). The agent parks the request in state and ends its turn instead of running the tool. The client UI shows the pending request. The user approves or denies via a [client action](./04-sdk.md#defining-client-actions). The agent re-emits the original call with the same `tool_call_id`, so the LLM sees one tool call and one result.

Example: [`examples/tool-approval`](https://github.com/substructureai/substructure/tree/main/examples/tool-approval).

## Modal agents (plan mode)

Run the agent in distinct phases: a planner that iterates on a TODO list with the user, then an executor that walks the list end-to-end. A `mode` field in state selects which phase is active; prompt, tools, and model all read from it. The client flips the mode with a client action, not a chat message, so the switch doesn't appear in the transcript. Clear the conversation history at the boundary if you want the executor to start with just the plan.

Example: [`examples/plan-mode`](https://github.com/substructureai/substructure/tree/main/examples/plan-mode).
