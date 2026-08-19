---
title: AG-UI
group: Frontends
---

AG-UI is a standard event protocol for agent chat UIs. The engine speaks it, so
frontends like CopilotKit and assistant-ui connect to an agent directly.

## Endpoint

Each agent has a run endpoint.

```
POST /api/channels/ag-ui/agents/{agent_id}/run
```

It takes a `RunAgentInput` and returns the AG-UI event stream over SSE.

A second endpoint, `.../connect`, sends the conversation as one snapshot for a
client that needs to catch up.

## Input

`RunAgentInput` carries the turn.

| Field | Meaning |
| --- | --- |
| `threadId` | The session. |
| `runId` | The turn ID. |
| `messages` | The client's view of the conversation. |
| `resume` | Answers to interrupts. |
| `tools`, `context`, `state`, `forwardedProps` | Supplied by the frontend. |

The engine applies the `resume` entries first. Then, if `messages` is not empty,
it submits a turn.

Send both in one input to change direction. The resume clears an open interrupt,
and the view branches the conversation somewhere else.

## Events

The stream carries AG-UI events.

| Event | Carries |
| --- | --- |
| `RUN_STARTED` / `RUN_FINISHED` | The turn. |
| `TEXT_MESSAGE_*` | Assistant text. |
| `REASONING_*` | Thinking. |
| `TOOL_CALL_*` | Tool calls and results. |
| `MESSAGES_SNAPSHOT` | The active branch. |

`RUN_FINISHED` also reports an interrupt when the agent paused.

While a thread has open interrupts, every run input must carry a `resume` for
each one. A missing or incomplete `resume` ends the run with `RUN_ERROR`, not an
HTTP error.

One run can resume and send messages at the same time. The messages continue the
turn the resume restarted. They do not open a second turn.

## Frontend tools and context

The engine turns `tools`, `context`, and `state` from `RunAgentInput` into the
decision's [client context](./150-client-tools.md#tools-from-the-client).

A frontend tool has `handler: "client"` and runs in the browser. When the model
calls one, the run ends so the browser can run the tool. The client then submits
again with the result.

## Examples

`node-hono-copilotkit` and `node-hono-assistant-ui` point the frontend at the
same endpoint.

```
${url}/api/channels/ag-ui/agents/${agentId}/run
```

Each declares a worker tool and a browser tool, to show both in one UI.

## Next

- [Client-side tools](./150-client-tools.md): the browser tools AG-UI carries.
- [Conversations](./120-conversations.md): the thread and tree behind a run.
- [Interrupts](./100-interrupts.md): how a run resumes one.
