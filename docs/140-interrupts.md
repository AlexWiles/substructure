---
title: Interrupts
group: Building agents
---

An interrupt is a saved pause. A worker stops the session to wait for a person.
A later resume gives the worker that person's answer. A paused session uses no
compute and survives a restart.

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

A person resumes it, and names the interrupt by id:

```jsonc
{ "type": "interrupt.resume", "interrupt_id": "int-1", "payload": { "approved": true } }
```

## Pausing

Return an `interrupt` action. `reason` is required. `payload` carries whatever
the person needs to see. The engine creates an `interrupt_id` if you omit one.

An interrupt attaches to the conversation head where it was raised. It pauses
that branch, not the whole session. Several interrupts can be open at once, on
the same branch or on different ones. An interrupt raised before any message
exists has nothing to attach to, so it pauses every branch. System pauses, such
as a budget stop, work this way.

The engine cancels the LLM calls in flight on the paused branch. Calls on other
branches keep running, and so do tools and sub-agents that already started.

## Resuming

An `interrupt.resume` input clears the interrupt by id. If that interrupt paused
the active branch, the worker receives an `interrupt.resumed` trigger with the
resume payload. Clearing an interrupt on a branch nobody uses sends nothing. An
old or repeated id does nothing.

## While paused

The engine refuses new messages on the paused branch until it resumes. It still
records work that ends an open call, but it holds the next decision until the
resume.

A paused turn is still the same turn. A pause emits no `turn.completed`, and a
resume starts no new turn. The `interrupt.resumed` trigger arrives inside the
turn that raised the interrupt, and events before and after the pause carry the
same `turn_id`.

A transport can still end its own unit of work. [AG-UI](./100-ag-ui.md) ends the
run at an interrupt, with `RUN_FINISHED` and an interrupt result, because a run
is one HTTP request. The resume opens a new run inside the same engine turn.

The rest of the tree stays live. If a client view edits an earlier message, it
branches below the interrupt's anchor and runs as normal. The user leaves the
paused question, and the interrupt stays open on the branch they left. If they
go back to that branch, the session is paused again, and the interrupt is still
there to answer or clear.

## Spec

```typescript
// action
{ type: "interrupt"; interrupt_id?: string; reason: string; payload?: unknown }

// trigger
{ type: "interrupt.resumed"; interrupt_id: string; payload?: unknown }

// client input
{ type: "interrupt.resume"; interrupt_id: string; payload?: unknown }
```

## Next

- [Durability](./110-durability.md): the engine saves the pause.
- [Deferred tools](./130-deferred-tools.md): wait on one call instead of the session.
- [Protocol](./150-protocol.md): the interrupt action, trigger, and input.
