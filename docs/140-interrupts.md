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
the human needs; the engine mints an `interrupt_id` when you omit one. The
session enters an interrupted state. LLM calls in flight are voided; tools and
sub-agents already running keep going.

## Resuming

An `interrupt.resume` input clears the interrupt and delivers an
`interrupt.resumed` trigger with its payload. Resume needs the interrupt's id
and an active turn. A stale or duplicate id is a no-op.

## While paused

New messages are refused until the session resumes. Work that settles an
in-flight call is still recorded, but its follow-on decision is held and
delivered only after the resume.

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
