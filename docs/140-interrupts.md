---
title: Interrupts
group: Building agents
---

An interrupt is a durable pause. A worker stops the session to wait for a
human, and a later resume hands it back the human's answer. A paused session
holds no compute and survives a restart.

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

A human resumes it, addressing the interrupt by id:

```jsonc
{ "type": "interrupt.resume", "interrupt_id": "int-1", "payload": { "approved": true } }
```

## Pausing

Return an `interrupt` action. `reason` is required; `payload` carries whatever
the human needs; the engine mints an `interrupt_id` when you omit one.

An interrupt is anchored to the conversation head where it was raised: it
parks that branch, not the whole session. Several interrupts can be open at
once, on the same or different branches. An interrupt raised before any
message exists has no anchor and parks every branch (the global case — how
system-level pauses like budget stops behave).

LLM calls in flight on the parked branch are voided; calls on other branches,
and tools and sub-agents already running, keep going.

## Resuming

An `interrupt.resume` input clears the interrupt by id. If the interrupt was
parking the active branch, the worker gets an `interrupt.resumed` trigger with
the resume payload; clearing an interrupt left behind on an abandoned branch
delivers nothing. A stale or duplicate id is a no-op.

## While parked

New messages extending the parked branch are refused until it resumes. Work
that settles an in-flight call is still recorded, but its follow-on decision
is held and delivered only after the resume.

The rest of the tree stays live. A client view that edits an earlier message
— branching off below the interrupt's anchor — dispatches normally: the user
walks away from the parked question and the interrupt stays open on the
abandoned branch. Switching back to that branch re-enters the parked state,
and the interrupt is still there to answer or clear.

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

- [Durability](./110-durability.md): the pause is persisted state.
- [Deferred tools](./130-deferred-tools.md): waiting on one call instead of the session.
- [Protocol](./150-protocol.md): the interrupt action, trigger, and input.
