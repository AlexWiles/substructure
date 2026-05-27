---
title: Core concepts
---

Before you build with Substructure, it helps to have a mental model of the pieces involved and how they fit together. This page is the glossary the rest of the docs assume you've read.

## The split: engine, workers, clients

Substructure is structured around three roles that talk to each other over HTTP:

- The **engine** is the Rust server that drives the agent loop. It owns the event log, schedules retries, makes LLM calls, and decides what should happen next. You can run it as Substructure Cloud, locally with `substructure local start`, or in-process via the embedded runtime.
- A **worker** is your code. It's an HTTP endpoint that the engine calls into whenever it needs you to make a decision: which tool to run, which sub-agent to delegate to, when to finish a turn. Workers are stateless and run wherever you deploy them.
- A **client** is whatever submits work to the engine: a backend service kicking off a turn, a browser streaming events into a chat UI, a script running a one-off task.

The key idea is that **the engine and the worker are separate processes**. The engine never executes your tool code; it just asks your worker what to do and acts on the response. That separation is what makes workers deployable to any serverless platform and what lets the engine survive worker restarts, redeploys, and crashes.

## Agents

An **agent** is a named entity that the engine routes decisions to, identified by a string `agentId` (`"weather-agent"`, `"todo"`, etc.). On the SDK side an agent is a chain of middleware: state management, system prompt, message history, tools, the LLM loop. On the engine side it's just a name that maps to a worker.

A single worker can host many agents. Clients pick one by `agentId` when they start a turn.

## Sessions

A **session** is a long-running context, identified by a `sessionId` (UUID). Think of it as a conversation: it has its own message history, its own durable state, its own event log. Two messages with the same `sessionId` are part of the same conversation; two with different IDs are unrelated.

Sessions never end on their own. They accumulate turns until a client stops sending or explicitly cancels them.

## Turns

A **turn** is one round of user input to final answer, identified by a `turnId`, always scoped to a session. A client starts a turn by calling `startTurn` with a message or action. The turn ends when the agent emits a `done` action, the engine puts the result on the event log, and the next turn can begin.

Inside one turn, many things can happen: the LLM is called, tools execute, sub-agents spawn, the LLM is called again with the results. All of that is the engine looping on your worker's decisions until the worker says "done."

## Decisions

A **decision** is one HTTP call from the engine to your worker. Every decision carries a **trigger** (what just happened) and the current **state** (either inline as a base64-encoded JSON blob, or empty if your worker loads state from its own database). Your worker responds with a list of **actions** (what to do next).

The engine carries out the actions, records what happened to the event log, and calls back with a new decision when there's something else for the worker to react to. This loop is the agent loop. The engine drives it; your worker decides what to do at each step.

### Triggers

The full set of triggers your worker may receive:

| Trigger | When the engine sends it |
| --- | --- |
| `user.message` | A client called `startTurn` with a chat message. |
| `client.action` | A client called `startTurn` with a typed action instead of a message. |
| `llm.response` | An LLM call completed. Payload includes the assistant message and any tool calls. |
| `llm.error` | An LLM call failed permanently (after retries). |
| `tool.execute` | The engine is asking your worker to run a tool. The SDK's `tools` middleware handles this and dispatches to your `execute` function. |
| `tool.result` | A tool finished, here is the result. |
| `sub_agent.turn.complete` | A child agent finished a turn; here's its output. |
| `sub_agent.error` | A child agent failed. |
| `interrupt.resumed` | A paused session was resumed by an external signal. |
| `stall` | Nothing has happened for a while; the worker has a chance to break the deadlock or finish. |

For most agents, the built-in middleware handles every trigger you'd see in practice. You only need to think about them when writing custom middleware.

### Actions

What your worker can return from a decision:

| Action | Effect |
| --- | --- |
| `call.llm` | Make an LLM request. Will produce an `llm.response` or `llm.error` trigger when it completes. |
| `call.tool` | Have the engine schedule a tool call. Produces a `tool.execute` trigger back at the worker. |
| `return.tool.result` | Return a result for a tool call the worker executed itself. |
| `return.tool.error` | Return an error for a tool call the worker executed itself. |
| `spawn.sub_agent` | Start a child session under a different agent. Produces `sub_agent.turn.complete` or `sub_agent.error` triggers. |
| `send.message` | Push a message into another session (handy for fan-out or notifying a parent). |
| `done` | Finish the turn. The `data` payload becomes the turn's result, returned from `client.turnResult(scope)`. |

A single decision can return multiple actions: for example, several `call.tool` actions in parallel, or a `send.message` followed by `done`.

## State

Each agent has a **state** object that persists across decisions within a session. This is where message history, tool-specific data, sub-agent tracking, and anything else the agent needs to remember between LLM calls lives.

You have three ways to hold state:

- **Wire state.** The SDK's `agent.jsonState()` middleware encodes state as a base64 JSON blob and ships it back and forth on every decision. Simple, no infrastructure required.
- **Your own database.** A custom middleware that loads state from your DB on the way in and saves it on the way out. Better for large state, sensitive data, or anything you want to query directly. See [Keep conversation state in your own database](./04-sdk.md#example-keep-conversation-state-in-your-own-database) in the SDK docs.
- **Hybrid.** Keep most state on the wire and pull individual slices out to your database. Useful when, for example, conversation history is small enough to ride along but a tool's working set is large. See [Hybrid wire and database state](./04-sdk.md#example-hybrid-wire-and-database-state).

State is logically per-session. Two sessions for the same user are independent unless you explicitly link them.

## Events

Every interesting thing that happens during a session is recorded as an **event**: messages sent, LLM calls requested and completed, tools invoked, sub-agents spawned, turns completed. The event log is append-only and durable. It's what makes the engine able to recover after a crash, what powers the debugging UI, and what `client.stream(scope)` is reading from when you tail a session in real time.

You can think of a session as the event log plus the derived state from replaying it.

## Identity

Every turn is submitted on behalf of an **identity**, an object with an `id` (your user id) and optional `metadata`. Identity is how the engine knows who a session belongs to. It flows through to your worker (`req.wire.identity.id`) so middleware and tools can scope behavior per user without trusting client-supplied data.

For browser clients, identity is baked into the short-lived token your backend mints; the browser can't change it.

## Sub-agents

A **sub-agent** is an agent another agent can delegate to, as if calling a tool. The parent emits a `spawn.sub_agent` action; the engine creates a child session, runs it to completion, and returns the result to the parent. The parent agent's session keeps the parent's history; the child's session keeps the child's. This lets you compose agents with clean isolation: a planner agent that delegates to specialist agents, a router that hands off to different worker pools, and so on.

## Where to next

Now that the vocabulary is settled:

- The [CLI](./03-cli.md) walkthrough provisions a cloud app, sets up the engine-to-worker webhook, and mints API keys for clients.
- The [SDK](./04-sdk.md) walkthrough builds an agent (middleware, tools, sub-agents), wraps it in a worker, and drives turns from backend or browser clients.
