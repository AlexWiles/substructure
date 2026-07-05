# tool-approval

Pause the agent before a sensitive tool runs, ask the client to approve
or deny it (optionally with a reason), then resume. To the LLM the
whole exchange looks like a normal tool call that just took a while.

## How it works

State rides the wire as `state` and carries two fields:

```ts
type State = {
    pendingCommand: { toolCallId: string; cmd: string } | null;
    approvalDecision: { approved: boolean; reason?: string } | null;
};
```

The default `toolLoop` drives the whole conversation — prompting,
running the tool, and continuing after the result. The agent is a custom
`decide` function that delegates to that loop and overrides only the two gate
triggers: parking a requested command, and resuming it on approval. The
approval decision itself is read inside the tool's `execute`.

**The tool** (`run_command`, built per decision, closing over `state`):
its `execute` reads `state.approvalDecision`. On a denial it returns
`{ exit_code: 1, stderr: "User denied this command. Reason: ..." }` as
its result; otherwise it runs the command via `spawnSync`. Because the
tool closes over `state`, the loop runs it normally on `effect.execute` —
the host shell actually executes here, so approve carefully.

**On the model's reply (`effect.settled`, `kind: "llm_call"`):** if the model called `run_command`, the `decide`
records the assistant turn, parks `{ toolCallId, cmd }` in
`state.pendingCommand`, and ends the turn with `done({ pendingCommand })`
instead of letting the loop run the call. The session is now waiting for
a decision. (Any other response falls through to `loop`, which finishes
it.)

**On `client.action approve_command`:** the `decide`:
1. Writes `{ approved, reason? }` to `state.approvalDecision`.
2. Clears `state.pendingCommand`.
3. Re-emits the parked call as `callTool({ toolCallId, name: "run_command",
   arguments })` — same `tool_call_id`, same arguments.

**Everything else** — prompting, actually running the approved command
(the tool's `execute`), and continuing after the result — is
`loop({ ...req, state })`.

Because the same `tool_call_id` is reused on the resume, the LLM
sees a single matched pair and never the pause. To it, the gate was
just a slow tool call, and the conversation transcript stays well-formed.

## Run

```sh
export OPENROUTER_API_KEY=sk-or-...
npm install

# Generate a session id, then reuse it across calls.
SESSION=$(uuidgen)

npx tsx index.ts $SESSION "List the files in this repo using shell."

# When a command is pending the CLI prints:
#   ⏸ awaiting approval for: ls -la
#      /approve  or  /deny [reason]  to continue

npx tsx index.ts $SESSION "/approve"
# or:
npx tsx index.ts $SESSION "/deny use a less verbose command"
```

The denial reason ("use a less verbose command") rides into the tool
result, the LLM reads it, and adapts on the next call.

## Adapt

- **Gate more tools**: broaden the check in the model-reply case
  (it currently looks for a `run_command` tool call) to match whichever
  tool names need approval. Generalize to a list or a predicate.
- **Different approval shapes**: pass extra fields in
  `approve_command.args` (timeout overrides, modified arguments, an
  approver id) and propagate them through `state.approvalDecision`.
- **Auto-approve trusted commands**: in the model-reply case, check
  the command against an allowlist before parking; re-emit the
  `callTool` directly if it matches.
- **Multiple pending approvals**: replace `pendingCommand` with a
  `Record<toolCallId, ...>` and match on the action when the
  client responds. Useful if multiple sensitive tools can fire in
  parallel.
