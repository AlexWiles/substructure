---
title: Core concepts
---

Before you build with Substructure, it helps to have a mental model of the pieces involved and how they fit together. This page is the glossary the rest of the docs assume you've read.

## The split: engine, workers, clients

Substructure is structured around three roles that talk to each other over HTTP:

- The **engine** is the Rust server that drives the agent loop. It owns the event log, schedules retries, makes LLM calls, and decides what should happen next. You can run it as Substructure Cloud, locally with `subs serve`, or in-process via the embedded runtime.
- A **worker** is your code. It's an HTTP endpoint that the engine calls into whenever it needs you to make a decision: which tool to run, which sub-agent to delegate to, when to finish a turn. Workers are stateless and run wherever you deploy them.
- A **client** is whatever submits work to the engine: a backend service kicking off a turn, a browser streaming events into a chat UI, a script running a one-off task.

The key idea is that **the engine and the worker are separate processes**. The engine never executes your tool code; it just asks your worker what to do and acts on the response. That separation is what makes workers deployable to any serverless platform and what lets the engine survive worker restarts, redeploys, and crashes.

## Agents

An **agent** is a named entity that the engine routes decisions to, identified by a string `agentId` (`"weather-agent"`, `"todo"`, etc.). On the SDK side an agent is `agent({ name, decide })`: `decide` is `toolLoop({ llm, instructions, tools, subAgents })` for the common tool/sub-agent loop, or your own function for full control over each decision. On the engine side it's just a name that maps to a worker.

A single worker can host many agents. Clients pick one by `agentId` when they start a turn.

## Sessions

A **session** is a long-running context, identified by a `sessionId` (UUID). Think of it as a conversation: it has its own message history, its own durable state, its own event log. Two messages with the same `sessionId` are part of the same conversation; two with different IDs are unrelated.

Sessions never end on their own. They accumulate turns until a client stops sending or explicitly cancels them.

## Turns

A **turn** is one round of user input to final answer, identified by a `turnId`, always scoped to a session. A client starts a turn by calling `startTurn` with a message or action. The turn ends when the agent emits a `done` action, the engine puts the result on the event log, and the next turn can begin.

Inside one turn, many things can happen: the LLM is called, tools execute, sub-agents spawn, the LLM is called again with the results. All of that is the engine looping on your worker's decisions until the worker says "done."

## Decisions

A **decision** is one HTTP call from the engine to your worker. Every decision carries a **trigger** (what just happened), the current **transcript** (the active conversation, on the wire as `messages`), and the current **state** (either inline as raw JSON, or empty if your worker loads state from its own database). Your worker responds with the updated **transcript** and a list of **actions** (what to do next).

The engine reconciles the returned transcript into the conversation tree, carries out the actions, records what happened to the event log, and calls back with a new decision when there's something else for the worker to react to. This loop is the agent loop. The engine drives it; your worker decides what to do at each step.

### Triggers

The full set of triggers your worker may receive:

Each trigger carries the new content and the current transcript; the worker folds the content into the transcript it returns (see Actions).

| Trigger | When the engine sends it |
| --- | --- |
| `client.messages` | The client proposed the conversation — a bare message, an AG-UI full view, an edit, a regenerate, all one shape. `messages` is the full proposed transcript (a bare message is materialized onto the active branch at delivery) and `messages[new_from..]` is the unrecorded news. The worker returns it (or amends it); the engine reconciles it into the tree. |
| `client.action` | A client called `startTurn` with a typed action instead of a message. |
| `tool.execute` / `llm.execute` | The engine is delegating a call to your worker: run a tool, or make an LLM call. `toolLoop` handles these and dispatches to the matching tool's `execute` or the worker-run model; a custom `decide` reacts to them directly. |
| `llm.finished` / `tool.finished` / `sub_agent.finished` | A call's final word: the model replied, or a tool/sub-agent finished. Fires as each one lands, so the transcript fills incrementally. The request's `calls` list says what's still in flight, so the worker prompts once no tool/sub-agent call is pending, without tracking the round itself. `ok` says whether the call succeeded. |
| `interrupt.resumed` | A paused session was resumed by an external signal. |

For most agents, `toolLoop` handles every trigger you'd see in practice. You only need to think about them when writing a custom `decide`.

### Actions

A decision returns a flat `messages` (the conversation as it should now be) plus a list of actions. The engine reconciles the transcript into the tree (the one place the tree is written) and carries out the actions:

| Action | Effect |
| --- | --- |
| `llm.call` | Make an LLM request with a prompt message list. Produces an `llm.finished` trigger when it completes. The prompt is separate from the transcript, so it can be shaped per call (compaction, injected context) without changing the record. |
| `tool.call` | Have the engine schedule a tool call, named by `id`. Produces a `tool.execute` trigger back at the worker. |
| `tool.result` / `llm.result` | Answer an execute with a success: a tool's `result` or a worker-run model's `response`. |
| `tool.error` / `llm.error` | Answer an execute with a failure. |
| `sub_agent.spawn` | Start a child session under a different agent. Its output (or error) comes back as a `sub_agent.finished` trigger when the child's turn completes. |
| `message.send` | Push a message into another session (handy for fan-out or notifying a parent). |
| `done` | Finish the turn. The `data` payload becomes the turn's result, returned from `client.turnResult(scope)`. |

A single decision can return multiple actions: for example, several `tool.call` actions in parallel, or a `message.send` followed by `done`.

## State

Across decisions in a session, two things persist: the **transcript** (the conversation tree, owned by the engine) and any **worker state** you choose to keep. There is no SDK-held tool state. Where your own state lives is a choice you make per agent:

- **Your own store.** Tools are pure functions that reach a store directly through the decision request, keyed by `request.session_id` (per conversation) or `request.identity.id` (per user). Best for large state, sensitive data, or anything you want to query directly; it never leaves your infrastructure. See [State](./04-sdk.md#state) in the SDK docs.
- **On the wire.** Keep small state on the decision envelope's `state` with a custom `decide`: the engine ships the state in as `req.state` (raw JSON) on every decision and persists whatever you return. Simple, no infrastructure required. See [State](./04-sdk.md#state).

State is logically per-session. Two sessions for the same user are independent unless you explicitly link them.

### State is branch-scoped

The conversation is a tree, and worker state follows it. The engine never overwrites your state in place: each write is recorded as a **version** anchored to the tree node that was the active head when it was written. The state a decision sees is resolved from the active branch — the newest version whose anchor lies on the path from the root to the current head.

On a linear conversation this is indistinguishable from a single mutable value. It matters when the tree forks (a client edits or regenerates an earlier message): versions written on the abandoned branch fall off the active path, so the new branch resolves to the state **as of the fork point** — memory, summaries, and instructions learned on the other branch don't leak in. A worker that *wants* to carry state across a fork simply submits it on the forking decision; the engine imposes neither semantic.

The rule of thumb: **conversational truth lives in state and forks with the branch; physical truth (what actually ran — every LLM call, tool result, and state version) lives in the event log and never forks.**

## Events

Every interesting thing that happens during a session is recorded as an **event**: messages sent, LLM calls requested and completed, tools invoked, sub-agents spawned, turns completed. The event log is append-only and durable. It's what makes the engine able to recover after a crash, what powers the debugging UI, and what `client.stream(scope)` is reading from when you tail a session in real time.

You can think of a session as the event log plus the derived state from replaying it.

`client.stream(scope, { tokens: true })` also interleaves transient `llm.token.delta` events when streaming is enabled on the agent's `llm` (they're off by default, so a plain `client.stream(scope)` yields only persisted events). Deltas are *not* persisted; they're a live side channel for progressive UI rendering. The canonical assistant text always arrives via the persisted `llm.call.completed` and `message.new` events that follow.

## Identity

Every turn is submitted on behalf of an **identity**, an object with an `id` (your user id) and optional `metadata`. Identity is how the engine knows who a session belongs to. It flows through to your worker (`request.identity.id`) so your tools and handlers can scope behavior per user without trusting client-supplied data.

For browser clients, identity is baked into the short-lived token your backend mints; the browser can't change it.

## Sub-agents

A **sub-agent** is an agent another agent can delegate to, as if calling a tool. The parent emits a `sub_agent.spawn` action; the engine creates a child session, runs it to completion, and returns the result to the parent. The parent agent's session keeps the parent's history; the child's session keeps the child's. This lets you compose agents with clean isolation: a planner agent that delegates to specialist agents, a router that hands off to different worker pools, and so on. See the [Sub-agents](./05-sub-agents.md) page for the full walkthrough.

