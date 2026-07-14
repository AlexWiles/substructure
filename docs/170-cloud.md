---
title: Cloud
group: Operations
---

Substructure cloud hosts the engine. A `subs`-managed app takes client traffic,
delivers each decision to your worker over a signed webhook, and by default
runs the LLM calls itself. `subs` is the control plane for that app.

## Sign in

```sh
subs login
```

Authenticates in your browser and stores a token under `~/.config/substructure`.

## Create an app

```sh
subs apps create my-bot
```

This prints the app id and its signing secret, shown once. Pin the app so later
commands need no `--app`:

```sh
subs link
```

## Point it at your worker

```sh
subs webhook set https://my-worker.example.com/agent
```

The engine now POSTs decisions to that URL and signs each with the app's secret,
an HMAC of the body sent as `X-Substructure-Signature`. Your worker verifies it
with the same secret.

## Secrets

Three secrets are in play. Two are set with `subs`.

| Secret | Purpose | Where it comes from |
| --- | --- | --- |
| Signing secret | Your worker verifies the engine's webhooks. | Printed by `apps create`; re-fetch with `subs webhook secret`, rotate with `subs webhook rotate-secret`. |
| Client API key | The bearer token your clients present to the app. | `subs keys create <label>`, printed once. |
| Provider key | Auth to Anthropic or OpenAI. | Not a `subs` secret. See below. |

`subs webhook secret` reprints the signing secret, and `subs keys create <label>`
mints a client key. Both write only the value to stdout, so you can pipe it
straight into your worker's or client's secret store.

### Provider keys

Where the model runs decides who holds the provider key. By default the hosted
engine runs the LLM against its own provider and bills the app, so you set no
key. If the agent declares `handler: "worker"`, the call runs in your worker,
which calls the provider with a key from its own environment. See
[LLMs](./50-llms.md).

## Observe

```sh
subs sessions list
subs sessions events <session-id> --stream
subs open
```

## Next

- [Quick start](./10-quick-start.md): build the worker this app calls.
- [CLI](./160-cli.md): the full command reference.
- [LLMs](./50-llms.md): where provider keys live.
- [Protocol](./150-protocol.md): the signed webhook the engine delivers.
