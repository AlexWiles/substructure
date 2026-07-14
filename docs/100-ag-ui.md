---
title: AG-UI
group: Frontends
---

AG-UI is a standard event protocol for agent chat UIs. The engine speaks it, so
frontends like CopilotKit and assistant-ui connect to an agent directly.

## Endpoint

Each agent is served at an AG-UI run endpoint:

```
POST /api/client/ag-ui/agents/{agent_id}/run
```

It takes a `RunAgentInput` and returns the AG-UI event stream over SSE. A
companion `.../connect` replays the conversation as a snapshot for a client that
is catching up.

## Input

`RunAgentInput` carries the turn.

| Field | Meaning |
| --- | --- |
| `threadId` | The session. |
| `runId` | The turn id. |
| `messages` | The client's conversation view. |
| `resume` | Interrupt answers, in place of messages. |
| `tools`, `context`, `state`, `forwardedProps` | Contributed by the frontend. |

A non-empty `resume` resumes an interrupt; otherwise `messages` submit a turn.

## Events

The stream is AG-UI events: `RUN_STARTED` and `RUN_FINISHED` around the turn,
`TEXT_MESSAGE_*` for assistant text, `REASONING_*` for thinking, `TOOL_CALL_*`
for tool calls and results, and `MESSAGES_SNAPSHOT` for the active branch.
`RUN_FINISHED` carries an interrupt outcome when the agent paused.

## Frontend tools and context

`tools`, `context`, and `state` from `RunAgentInput` become the decision's
[client context](./90-client-tools.md#client-contributed-tools). Frontend tools
are declared `handler: "client"` and run in the browser; a call to one ends the
run so the browser can execute it, then the client resubmits with the result.

## Examples

`node-hono-copilotkit` and `node-hono-assistant-ui` point the frontend at the
same endpoint:

```
${url}/api/client/ag-ui/agents/${agentId}/run
```

Each declares a worker tool and a browser tool, showing both sides in one UI.

## Next

- [Conversations](./70-conversations.md): the thread and tree behind a run.
- [Client-side tools](./90-client-tools.md): the browser tools AG-UI carries.
- [Interrupts](./140-interrupts.md): the resume path a run drives.
