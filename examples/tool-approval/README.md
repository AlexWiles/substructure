# tool-approval

Pause the agent before a sensitive tool runs, ask the client to approve
or deny it (optionally with a reason), then resume. To the LLM the
whole exchange looks like a normal tool call that just took a while.

## How it works

State adds two fields:

```ts
type ApprovalState = {
    pendingCommand: { toolCallId: string; cmd: string } | null;
    approvalDecision: { approved: boolean; reason?: string } | null;
};
```

The `approvalGate` middleware sits *outside* the `agent.tools`
middleware so it can see the `call.tool` actions tools emits.

**On the way out (any trigger):** if a `call.tool` action targets
`run_command`, the middleware:
1. Parks the request in `state.pendingCommand`.
2. Removes the `call.tool` from the action list.
3. Drops any pending `call.llm` so the turn pauses cleanly.
4. Appends a `done` action with the pending command in its data.

The turn ends. The session is now waiting for a decision.

**On a `client.action approve_command` trigger:** the middleware:
1. Writes `{ approved, reason? }` to `state.approvalDecision`.
2. Clears `state.pendingCommand`.
3. Re-emits the original `call.tool` action — same `tool_call_id`,
   same arguments.

The runtime dispatches the re-emitted call as a normal `tool.execute`.
`run_command`'s body reads `state.approvalDecision`: on approve it
runs the command; on deny it returns `{ exit_code: 1, stderr: "User
denied this command. Reason: ..." }`. Either way the result rides
back through tool history → triggers `call.llm` → the LLM sees a
matched tool_call_id pair and continues.

Because the same `tool_call_id` is reused on the resume, the LLM
never sees the pause. The conversation transcript stays well-formed.

## Run

```sh
export OPENROUTER_API_KEY=sk-or-...
pnpm install

# Generate a session id, then reuse it across calls.
SESSION=$(uuidgen)

pnpm tsx index.ts $SESSION "List the files in this repo using shell."

# When a command is pending the CLI prints:
#   ⏸ awaiting approval for: ls -la
#      /approve  or  /deny [reason]  to continue

pnpm tsx index.ts $SESSION "/approve"
# or:
pnpm tsx index.ts $SESSION "/deny use a less verbose command"
```

The denial reason ("use a less verbose command") rides into the tool
result, the LLM reads it, and adapts on the next call.

## Adapt

- **Gate more tools**: change the filter in `approvalGate` to match
  whichever tool names need approval. Generalize to a list or a
  predicate.
- **Different approval shapes**: pass extra fields in
  `approve_command.args` (timeout overrides, modified arguments, an
  approver id) and propagate them through `state.approvalDecision`.
- **Auto-approve trusted commands**: in `approvalGate`, check the
  command against an allowlist before parking; emit it directly if
  it matches.
- **Multiple pending approvals**: replace `pendingCommand` with a
  `Record<toolCallId, ...>` and match on the action id when the
  client responds. Useful if multiple sensitive tools can fire in
  parallel.
