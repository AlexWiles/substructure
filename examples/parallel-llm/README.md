# parallel-llm

Fan out several `llm.call` actions in one decision, layered over the stock
`toolLoop`. A thin wrapper owns the fan-out: on the user's message it issues one
`llm.call` per "lens", each named by its own `id`; each settles independently as
its own `llm.finished` trigger, serialized on the decision
stream (a worker never sees two decisions at once) but arriving in completion
order. Once every lens has landed, the wrapper hands the enriched question to the
wrapped `toolLoop`, which synthesizes the final answer — and every trigger the
wrapper doesn't recognize delegates straight to the loop, so tools, the loop's
own LLM settles, and `done` all keep working.

Two agents:

- **`fanout`** — engine-handled lenses (`handler: "server"`). The engine calls
  its provider for each; the worker just issues them and folds the settles. The
  engine executes a session's calls one at a time (a deliberate bound on
  per-session provider pressure), so this decouples your loop rather than
  shrinking wall-clock time — and each call's retry deadline keeps ticking while
  it waits behind its siblings, so give fanned-out calls generous `timeout_secs`.
- **`deferred-fanout`** — worker-handled lenses (`handler: "worker"`). Each call
  arrives as an `llm.execute`; the worker starts it in the background, returns
  the decision immediately with no actions (so the next `llm.execute` promotes
  at once and the calls overlap), and settles each out-of-band with
  `settleEffect` when it finishes. This is the path to true wall-clock overlap.
  `runModel` is a stub standing in for your real provider call, and the loop's
  own synthesis call runs on the worker too, so this flavor needs no API key.

Ids must be **fresh per logical call** — reusing a pending/completed id is an
idempotent no-op (that's what makes decision redelivery and retry-after-interrupt
safe), so an accidentally reused id silently loses the call.

Two equivalent ways to detect drain for a single fan-out. `d.calls` lists the
calls still in flight, frozen on each decision at commit time and delivered in
completion order — so it empties exactly on the last settle, after every sibling
has been folded. This example instead tracks the ids it issued, which is also
correct and stays correct if you ever spread a fan-out across several turns (where
"nothing in `effects` right now" isn't the same as "the calls I care about are all
back").

This example runs the runtime in-process (`SubstructureEmbedded.create`), so a
single file drives the whole demo.

## Run

```sh
npm install

# engine-handled fan-out (calls the real provider)
export OPENROUTER_API_KEY=sk-or-...
npm start

# worker-handled deferred fan-out (settles via settleEffect; fully stubbed, no key)
npm start deferred
```
