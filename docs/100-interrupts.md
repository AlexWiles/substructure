---
title: Interrupts
group: Building agents
---

An interrupt is a saved pause. A worker stops the session to wait for a person.

A paused session uses no compute and survives a restart. A later resume gives
the worker that person's answer.

## Example

The worker pauses before a sensitive action, then acts on the answer.

```javascript title="server.mjs"
function decide({ trigger, proposed }) {
    if (trigger.type === "tool.execute" && trigger.name === "send_email") {
        return { actions: [{ type: "interrupt", reason: "confirm", payload: { message: "Send the email?" } }] };
    }

    if (trigger.type === "interrupt.resumed") {
        return trigger.payload?.approved
            ? { actions: [{ type: "tool.result", result: "sent" }] }
            : { actions: [{ type: "tool.error", error: "declined" }] };
    }

    return proposed;
}
```

A person resumes it by id.

```jsonc
{ "type": "interrupt.resume", "interrupt_id": "int-1", "payload": { "approved": true } }
```

In Slack, an interrupt with a `message` in its payload posts buttons. See
[Slack](./130-slack.md#interrupt-prompts).

## Pausing

Return an `interrupt` action. `reason` is required. `payload` carries whatever
the person needs to see. The engine creates an `interrupt_id` if you omit one.

An interrupt attaches to the conversation head where it was raised. It pauses
that branch. Several interrupts can be open at once, on the same branch or on
different ones.

An interrupt raised before any message exists pauses every branch. System
pauses, such as a budget stop, work this way.

The engine cancels the model calls in flight on the paused branch. Calls on
other branches keep running, and so do tools and sub-agents that already
started.

## Resuming

An `interrupt.resume` input clears the interrupt by id.

If that interrupt paused the active branch, the worker receives an
`interrupt.resumed` trigger with the resume payload. Clearing an interrupt on a
branch nobody uses sends nothing. An old or repeated id does nothing.

## While paused

The engine refuses new messages on the paused branch. It still records work that
ends an open call, and holds the next decision until the resume.

A paused turn is still the same turn. A pause emits no `turn.completed`, and a
resume starts no new turn. Events before and after the pause carry the same
`turn_id`.

A frontend can still end its own unit of work. [AG-UI](./140-ag-ui.md) ends the
run at an interrupt, because a run is one HTTP request. The resume opens a new
run inside the same engine turn.

The rest of the tree stays live. A client view that edits an earlier message
branches below the interrupt's anchor and runs as normal. The interrupt stays
open on the branch they left.

## Spec

```typescript
// action
{ type: "interrupt", interrupt_id?: string, reason: string, payload?: unknown }

// action, to clear one yourself
{ type: "interrupt.resolve", interrupt_id: string, payload?: unknown }

// trigger
{ type: "interrupt.resumed", interrupt_id: string, payload?: unknown }

// client input
{ type: "interrupt.resume", interrupt_id: string, payload?: unknown }
```

## Next

- [Slack](./130-slack.md#interrupt-prompts): approval buttons in a thread.
- [Async tools](./110-async-tools.md): wait on one call instead of the session.
- [Durability](./200-durability.md): the engine saves the pause.
