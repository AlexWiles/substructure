---
title: Human-in-the-loop
---

A sensitive tool call should sometimes pause for human approval before it runs. Substructure has no special primitive for this. You build it from ordinary pieces: a middleware that intercepts the tool call, a [client action](../04-sdk.md#defining-client-actions) that resumes it, and a `tool_call_id` reused on resume so the LLM's transcript stays well-formed.

The full runnable code is at [`examples/tool-approval`](https://github.com/substructureai/substructure/tree/main/examples/tool-approval). This page walks through how the pieces fit so you can adapt the pattern.

## When to reach for it

- **Sensitive operations.** Anything that spends money, mutates production data, sends a message, or shells out should ask first.
- **Confirmation flows.** "About to send this email, ok?", "About to delete these 12 rows, ok?", with the option to deny or amend.
- **Streaming review.** Show what the agent is about to do, let the user edit or veto, then continue.

If the gate is deterministic (an allowlist of commands, a numeric threshold), enforce it in the tool body and skip the pattern. Reach for this when the decision needs human judgment.

## The pattern, end to end

When the LLM emits a tool call for a sensitive tool, a gate middleware intercepts it on the way out of the chain, parks the request in state, and ends the turn with a `done` action carrying the pending request. The client sees a normal turn completion whose data describes "an approval is pending"; its UI surfaces a prompt to the user.

The user approves or denies. The client resumes by calling a client action on the same session. The action handler records the decision in state and re-emits the original `call.tool` with the **same `tool_call_id`**. The engine executes it as a normal `tool.execute`; the tool reads the decision from state and either does the real work or returns a denial; the LLM sees a matched `(tool_call_id, tool_result)` pair as if the call had simply taken a while.

That matched `tool_call_id` is the key invariant. It's what keeps the LLM's transcript well-formed across the pause.

## State

Two slots own the handoff:

```ts
type ApprovalState = {
  pendingCommand: { toolCallId: string; cmd: string } | null;
  approvalDecision: { approved: boolean; reason?: string } | null;
};
```

`pendingCommand` is what the client needs to render. `approvalDecision` is what the tool reads on resume.

## The gate middleware

Owns the slice and runs on the way out of `next(req)`. If any returned action is a `call.tool` for the sensitive tool, pull it out, park it in state, drop any pending `call.llm`, and end the turn:

```ts
const approvalGate = middleware<ApprovalState>({
  state: { pendingCommand: null, approvalDecision: null },
  handler: async (req, next) => {
    const result = await next(req);
    const pending = result.actions.find(
      (a) => a.type === "call.tool" && a.name === "run_command",
    );
    if (!pending) return result;

    req.state.pendingCommand = {
      toolCallId: pending.tool_call_id,
      cmd: JSON.parse(pending.arguments).cmd,
    };
    return {
      ...result,
      actions: [{ type: "done", data: { pendingCommand: req.state.pendingCommand } }],
    };
  },
});
```

The gate sits *outside* `agent.tools` in the chain so it can see the `call.tool` actions the tools middleware emits.

## The approval action

A `client.action` handler that records the decision and re-emits the parked `call.tool` with the same id:

```ts
const approveCommand = agent.action({
  name: "approve_command",
  state: approvalGate,
  handler: (args: { approved: boolean; reason?: string }, state) => {
    const pending = state.pendingCommand;
    if (!pending) return [{ type: "done", data: "no pending command" }];

    state.approvalDecision = args;
    state.pendingCommand = null;
    return [
      {
        type: "call.tool",
        tool_call_id: pending.toolCallId,
        name: "run_command",
        arguments: JSON.stringify({ cmd: pending.cmd }),
        handler: "worker",
        retry: DEFAULT_RETRY,
      },
    ];
  },
});
```

Returning a `WorkerAction[]` short-circuits the rest of the chain, including the gate. That short-circuit is what lets the re-emitted `call.tool` bypass the gate it would otherwise hit.

## The tool body

Reads `state.approvalDecision`. On denial, returns a denial the LLM can react to:

```ts
const runCommand = agent.tool({
  name: "run_command",
  description: "Run a shell command. Requires user approval.",
  parameters: { type: "object", properties: { cmd: { type: "string" } }, required: ["cmd"] },
  state: approvalGate,
  execute: (args, state) => {
    const decision = state.approvalDecision;
    state.approvalDecision = null;
    if (decision && !decision.approved) {
      return { denied: true, reason: decision.reason };
    }
    return runShellCommand(JSON.parse(args).cmd);
  },
});
```

The denial returns as a normal tool result; the LLM reads the reason on the next decision and adapts.

## Chain order

```ts
agent({ id: "assistant" })
  .use(agent.jsonState())
  .use(agent.systemMessage("..."))
  .use(agent.messageHistory())
  .use(agent.actions([approveCommand]))   // above the gate
  .use(approvalGate)                      // above tools
  .use(agent.tools([runCommand]))
  .use(agent.llmLoop({ request: { /* ... */ } }));
```

Two rules carry the whole pattern:

1. `agent.actions` sits **above** `approvalGate`. The action handler short-circuits everything below it; if the gate sat above the action handler, the resumed `call.tool` would pause again forever.
2. `approvalGate` sits **above** `agent.tools`. The gate only sees `call.tool` actions on the way back up after the tools middleware produces them.

## What the client sees

A paused turn produces a normal `turn.completed` event whose `data` carries `{ pendingCommand }`. That's the whole signal: no special event type, no back-channel.

To resume, call `startTurn` on the same `sessionId` with `payload: { type: "action", name: "approve_command", args: { approved, reason? } }`. The agent picks up where it paused.

## Adapting

- **Gate more tools.** Make the gate's check a set or a predicate over tool name. Same parking logic.
- **Auto-approve trusted requests.** Inspect the arguments before parking; if they match an allowlist, leave the `call.tool` in place and let it run unattended.
- **Richer decisions.** Add fields to `approve_command.args` (a timeout override, a modified argument, an approver id) and propagate them through `state.approvalDecision` into the tool body.
- **Multiple pending approvals.** Replace `pendingCommand` with `Record<toolCallId, ...>` and match on the id when the client responds. Useful when several sensitive tools can fire in parallel.
