# deferred-tool

A tool whose result arrives out-of-band via `settleEffect`. Declaring it with
`deferred: true` makes `execute` a **kick-off**: it runs to start the real work
(webhook, long job, human approval), its return value is ignored, and the loop
emits **no result action** — so the engine leaves the tool call pending. When
the work finishes, `settleEffect` delivers the result and the loop resumes as
if the tool had returned synchronously. If the kick-off itself throws, the
loop reports it as an `effect.error` like any other tool failure.

This example runs the runtime in-process (`SubstructureEmbedded.create`), so a single
file drives the whole demo.

## Run

```sh
export OPENROUTER_API_KEY=sk-or-...
pnpm install
pnpm start
```

You'll see `tool.call.requested` fire immediately, a 3-second pause, then
`tool.call.completed` followed by the final assistant message.
