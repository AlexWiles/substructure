# ai-sdk-example

Runs LLM calls inside the worker with OpenRouter and streams token deltas
back through Substructure.

The call is made with the Vercel AI SDK (`ai`) via `@openrouter/ai-sdk-provider`,
so the worker forwards `streamText().fullStream` parts to `emitDelta` with no
hand-rolled SSE parsing. Swap providers by changing the `model` line.

The worker still receives signed webhooks from Substructure. When the engine
requests `text/event-stream`, the SDK worker response streams transient
`llm.token.delta` frames and finishes with the normal decision result.

## Run

In one terminal, start a local Substructure server pointed at this worker:

```sh
export OPENROUTER_API_KEY=sk-or-...
substructure start --dev --port 9000 --worker-url http://localhost:3030
```

In another terminal, start the worker:

```sh
export OPENROUTER_API_KEY=sk-or-...
pnpm install
pnpm start
```

In a third terminal, submit a turn and watch token deltas stream:

```sh
pnpm client
```

Set `OPENROUTER_MODEL` to override the default model.
