# deferred-tool

A tool that calls `ctx.defer()` and completes its result out-of-band via
`submitToolCallResult`. This is the pattern for async tool calls: kick off
real work (webhook, long job, human approval) inside `execute`, return
`ctx.defer()`, and submit the result when it's ready.

When `execute` returns `ctx.defer()`, the `tools` middleware emits no
`return.tool.result` for that call. The worker submits zero actions for
that `tool.execute` trigger, and the engine leaves the tool call pending
until the result lands.

This example runs the runtime in-process (`sub.embedded`), so a single
file drives the whole demo.

## Run

```sh
export OPENROUTER_API_KEY=sk-or-...
pnpm install
pnpm start
```

You'll see `tool.call.requested` fire immediately, a 3-second pause, then
`tool.call.completed` followed by the final assistant message.
