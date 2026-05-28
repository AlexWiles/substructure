---
title: Sub-agents
---

A sub-agent is an agent another agent can delegate to as if calling a tool. The parent's LLM sees a "tool" with the sub-agent's name and a single `message: string` parameter. Calling it spawns a child session that runs to completion in its own loop and returns a single result.

## Defining a sub-agent

Sub-agents are ordinary agents. There's nothing special about how they're built:

```ts
const weatherAgent = agent({ id: "weather" })
  .use(agent.jsonState())
  .use(agent.messageHistory())
  .use(agent.systemMessage("Weather assistant. Look up the weather. Be concise."))
  .use(agent.tools([getWeather]))
  .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-5" } }));
```

What makes it a sub-agent is how a parent uses it.

## Delegating from a parent

The parent declares the children it can delegate to with `agent.subAgents`:

```ts
const assistant = agent({ id: "assistant" })
  .use(agent.jsonState())
  .use(agent.messageHistory())
  .use(agent.systemMessage("Helpful assistant. Delegate weather questions to the weather agent."))
  .use(agent.subAgents({ agents: [weatherAgent] }))
  .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-5" } }));
```

Then deploy both agents in the same worker:

```ts
const worker = sub.worker({ agents: [assistant, weatherAgent] });
```

That's the whole setup. From the parent's LLM, `weather` looks like a tool with a single string argument; the SDK derives the tool definition automatically from the agent's id.

## Under the hood

When the parent's LLM emits a tool call for a sub-agent name, the `subAgents` middleware does four things:

1. Generates a new `sessionId` for the child.
2. Emits a `spawn.sub_agent` action with the child's session and agent ids.
3. Emits a `send.message` action that delivers the LLM's `message` argument to the child as a user message.
4. Records `childSessionId -> { toolCallId, agentName }` in `state.subAgentTracker` so it can correlate the result later.

The engine creates the child session, runs it to completion, and sends a trigger back to the parent:

- `sub_agent.turn.complete` if the child finished with a `done` action. The middleware converts this into a `tool.result` trigger with the child's output as the content, on the original tool call id.
- `sub_agent.error` if the child failed past its retries. Same path, but `is_error: true` and the error message as content.

From the parent LLM's perspective, the whole thing looks like a tool call that took a while and returned a string. It doesn't see the child's intermediate steps, tool calls, or LLM responses.

The child runs in its own session with its own state, tools, sub-agents (recursively), model, and middleware stack. Its identity is inherited from the parent's session.

## The tool the LLM sees

For each agent passed to `subAgents`, the LLM sees:

```json
{
  "type": "function",
  "function": {
    "name": "<agentId>",
    "description": "Delegate to <agentId>",
    "parameters": {
      "type": "object",
      "properties": {
        "message": { "type": "string", "description": "The message to send to the agent" }
      },
      "required": ["message"]
    }
  }
}
```

The agent id is the tool name and there's exactly one parameter, `message`. Pick agent ids the LLM can interpret well (`weather`, `book_flight`, `summarize_pr`), and rely on the parent's system prompt to spell out when each one should be used.

If you want a richer interface than a single string, define a regular `agent.tool` instead. The current shape keeps the contract uniform and the parent agnostic about the child's internals.

## Parallel fan-out

The LLM can return multiple tool calls in a single response. The `subAgents` middleware spawns one child session per call without waiting; the engine processes the spawns independently, so children run concurrently and their results come back to the parent as each finishes.

If the user asks "weather in SF, NYC, and Tokyo" and the LLM emits three `weather` calls in one response, three child sessions run in parallel and the parent's LLM gets all three results in the next decision. No special wiring required.

## Same agent, two roles

Sub-agent invocation doesn't use a different code path from a normal `startTurn`. Both go through the same agent id and produce a session with its own history. An agent you wrote to be called by other agents can also be called directly by a client:

```ts
const weatherAgent = agent({ id: "weather" }).use(/* ... */);

const assistant = agent({ id: "assistant" })
  .use(/* ... */)
  .use(agent.subAgents({ agents: [weatherAgent] }))
  .use(agent.llmLoop({ request: { model: "anthropic/claude-sonnet-4-5" } }));

sub.worker({ agents: [assistant, weatherAgent] });
```

A weather-only client can `startTurn({ agentId: "weather", ... })` directly, and the assistant can delegate to it as a sub-agent. Same code. Useful when you want a small focused agent to be both a building block and a standalone product surface.

## Errors, retries, and cost

- **Retries** apply to the child session's worker decisions, not to the spawn itself. If the child's worker crashes or returns an error, the engine retries that decision per the child's retry policy. Pass `retry` to `subAgents({ agents, retry })` to override the default for spawn-related retries.
- **Errors** that exhaust retries surface to the parent as a tool error (`is_error: true`) with the error string as content. The parent's LLM sees it, gets a chance to recover, and the parent's session stays alive.
- **Cost and token usage** for the child accumulate on the child session, then roll up into `sub_agent_cost` and `sub_agent_token_usage` on the parent's session state. The `TurnResult` your top-level client sees includes the full rolled-up cost.

## Observing a session

You can stream events from the parent session to watch the whole tree. The interesting events for sub-agents:

| Event | What it means |
| --- | --- |
| `sub_agent.requested` | Parent asked the engine to spawn the child. |
| `sub_agent.started` | Child session was created. |
| `sub_agent.turn_completed` | Child finished a turn; the result will land back at the parent as a `tool.result` trigger. |
| `sub_agent.errored` | Child failed past retries. |

The child has its own session id, so if you want to drill into what it actually did, stream that session directly with `client.stream({ sessionId: childSessionId })`. The `ancestry` field on the child session lets you walk back up the tree.

