---
title: CLI
---

The `substructure` CLI is how you provision and operate Substructure from the terminal. It can also run a local server for development.

## Install

```sh
npm i -g @substructure.ai/cli
```

Supported platforms: macOS (arm64, x64) and Linux (arm64, x64). The npm package ships a thin Node.js wrapper that loads a compiled Rust binary for your platform.

Verify the install:

```sh
substructure --help
```

The commands are flat. `serve` runs a Substructure server on your machine; everything else — `login`, `link`, `orgs`, `apps`, `keys`, `webhook`, `sessions`, `open` — manages a server you talk to over HTTP.

Those management commands aren't cloud-only: they work against [Substructure Cloud](https://app.substructure.ai) or a local server, depending on which one you point them at. By default they target the cloud; pass `--url` (or set `$SUBS_API_URL`) to target something else. See [Targeting a server](#targeting-a-server).

## Walkthrough

A typical first session against the cloud looks like this.

### 1. Log in

```sh
substructure login
```

This kicks off an OAuth device flow: a verification URL opens in your browser and the CLI waits for you to approve. The resulting bearer token is written to `~/.config/substructure/credentials.toml`.

Pass `--no-browser` and the CLI will print the URL instead.

Check who you're logged in as at any point:

```sh
substructure whoami
```

And later, when you want to clear credentials:

```sh
substructure logout
```

### 2. Pick an org and app

Against the cloud, each command runs against an org and (usually) an app. You can either pass them explicitly each time with `--org` and `--app`, or pin them to your project directory once (a local server supplies them automatically — see [Targeting a server](#targeting-a-server)):

```sh
substructure link
```

This walks you through picking an org and app interactively, then writes a `substructure.toml` to the current directory. Subsequent commands run from this directory will use those values by default.

To see what orgs you belong to:

```sh
substructure orgs list
```

### 3. Create or pick an app

If you don't have an app yet:

```sh
substructure apps create my-app
```

Other things you can do with apps:

```sh
substructure apps list           # list apps in the current org
substructure apps show           # show details for the linked app
substructure apps rename         # rename an app
substructure apps delete         # delete an app (owner only)
```

### 4. Point Substructure at your worker

A worker is the HTTP endpoint where your agent logic lives. Tell Substructure where to call it:

```sh
substructure webhook set https://your-worker.example.com
```

The webhook is signed with a secret you set in your worker's environment. Print it:

```sh
substructure webhook secret
```

This command prints just the raw secret, so it pipes cleanly into deploy tools:

```sh
substructure webhook secret | wrangler secret put SIGNING_SECRET
```

To temporarily stop deliveries without losing config:

```sh
substructure webhook disable
```

If a secret is ever leaked, rotate it (owner only):

```sh
substructure webhook rotate-secret
```

### 5. Mint API keys for clients

Clients use API keys to submit turns. Issue one:

```sh
substructure keys create demo
```

Like the webhook secret, this prints only the key, so you can capture it directly:

```sh
export SUBSTRUCTURE_API_KEY=$(substructure keys create demo)
```

List and revoke as needed:

```sh
substructure keys list
substructure keys revoke <KEY_ID>
```

### 6. Debug a running agent

Once turns are flowing, you can tail what's happening from the terminal.

List recent debug sessions:

```sh
substructure sessions list
substructure sessions list --agent-id weather-agent
```

Stream events from a session as they arrive (Ctrl-C to stop):

```sh
substructure sessions events <SESSION_ID>
```

By default this replays full history then continues live. Pass `--from <N>` to skip to a specific event index.

To jump to the admin UI for an app:

```sh
substructure open
```

## Local server

For local development you can run the server on your machine instead of using the cloud:

```sh
substructure serve \
  --dev \
  --provider openrouter \
  --worker-url http://localhost:4444
```

What the flags do:

- `--dev` disables client and worker authentication. Never use this in production.
- `--provider openrouter` selects the LLM provider. Reads `OPENROUTER_API_KEY` from the environment.
- `--worker-url` pre-registers an HTTP worker at startup, so you don't need to register one through the API.

Other useful flags:

- `--host` (default `127.0.0.1`) and `--port` (default `8080`) control where the server binds.
- `--db` (default `data.db`) sets the SQLite file used for durable state.
- `--signing-secret` sets the worker signing secret explicitly. If omitted, one is generated and printed.

## Targeting a server

A local server speaks the same API as the cloud, so the management commands work against it too — you just point them somewhere other than the default. The target URL resolves in this order: the `--url` flag, then `$SUBS_API_URL`, then a `url` pinned in `substructure.toml`, then the cloud default (`https://api.substructure.ai`).

```sh
substructure sessions list --url http://localhost:8080
substructure webhook set https://your-worker.example.com --url http://localhost:8080
```

**Auth.** Commands send a bearer token resolved as `$SUBS_API_TOKEN`, then the token `login` stored for that URL. A `serve --dev` server ignores auth, so no token is needed; for a self-hosted server with auth enabled, set the env var:

```sh
SUBS_API_TOKEN=… substructure sessions list --url https://subs.internal
```

`login` is URL-aware — the token it saves is scoped to the server you logged into — so you can stay logged in to several servers at once (e.g. cloud and a staging deploy) and each command only ever sends the token for its target.

**Orgs and apps.** A local server is single-tenant: it has one org and one app, so `--org`/`--app` are optional against it and you're never prompted. Commands that only make sense for the multi-tenant cloud (`apps create`, `keys`, …) return an error against a local server.

## Global flags

These apply to any management command:

- `--org <ID>` / `--app <ID>`: target a specific org or app, overriding `substructure.toml`. Optional against a local server.
- `--config <PATH>`: use a `substructure.toml` from a custom path.
- `--credentials <PATH>`: use a credentials file from a custom path.
- `--url <URL>`: override the API URL (also reads `$SUBS_API_URL`). The bearer token is read from `$SUBS_API_TOKEN`.
- `--json`: emit machine-readable JSON instead of formatted tables.
- `--no-interaction` (`-n`): never prompt; fail if input would be required. Useful in scripts and CI.

