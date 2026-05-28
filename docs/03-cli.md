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

The CLI is organized into two top-level groups:

- `substructure cloud` for provisioning and managing [Substructure Cloud](https://app.substructure.ai).
- `substructure local` for running a Substructure server on your machine.

## Cloud walkthrough

Most of what you'll do day-to-day lives under `substructure cloud`. A typical first session looks like this.

### 1. Log in

```sh
substructure cloud login
```

This kicks off an OAuth device flow: a verification URL opens in your browser and the CLI waits for you to approve. The resulting bearer token is written to `~/.config/substructure/credentials.toml`.

Pass `--no-browser` and the CLI will print the URL instead.

Check who you're logged in as at any point:

```sh
substructure cloud whoami
```

And later, when you want to clear credentials:

```sh
substructure cloud logout
```

### 2. Pick an org and app

Every cloud command runs against an org and (usually) an app. You can either pass them explicitly each time with `--org` and `--app`, or pin them to your project directory once:

```sh
substructure cloud link
```

This walks you through picking an org and app interactively, then writes a `substructure.toml` to the current directory. Subsequent commands run from this directory will use those values by default.

To see what orgs you belong to:

```sh
substructure cloud orgs list
```

### 3. Create or pick an app

If you don't have an app yet:

```sh
substructure cloud apps create my-app
```

Other things you can do with apps:

```sh
substructure cloud apps list           # list apps in the current org
substructure cloud apps show           # show details for the linked app
substructure cloud apps rename         # rename an app
substructure cloud apps delete         # delete an app (owner only)
```

### 4. Point Substructure at your worker

A worker is the HTTP endpoint where your agent logic lives. Tell Substructure where to call it:

```sh
substructure cloud webhook set https://your-worker.example.com
```

The webhook is signed with a secret you set in your worker's environment. Print it:

```sh
substructure cloud webhook secret
```

This command prints just the raw secret, so it pipes cleanly into deploy tools:

```sh
substructure cloud webhook secret | wrangler secret put SIGNING_SECRET
```

To temporarily stop deliveries without losing config:

```sh
substructure cloud webhook disable
```

If a secret is ever leaked, rotate it (owner only):

```sh
substructure cloud webhook rotate-secret
```

### 5. Mint API keys for clients

Clients use API keys to submit turns. Issue one:

```sh
substructure cloud keys create demo
```

Like the webhook secret, this prints only the key, so you can capture it directly:

```sh
export SUBSTRUCTURE_API_KEY=$(substructure cloud keys create demo)
```

List and revoke as needed:

```sh
substructure cloud keys list
substructure cloud keys revoke <KEY_ID>
```

### 6. Debug a running agent

Once turns are flowing, you can tail what's happening from the terminal.

List recent debug sessions:

```sh
substructure cloud sessions list
substructure cloud sessions list --agent-id weather-agent
```

Stream events from a session as they arrive (Ctrl-C to stop):

```sh
substructure cloud sessions events <SESSION_ID>
```

By default this replays full history then continues live. Pass `--from <N>` to skip to a specific event index.

To jump to the admin UI for an app:

```sh
substructure cloud open
```

## Local server

For local development you can run the server on your machine instead of using the cloud:

```sh
substructure local start \
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

## Global flags for cloud commands

These apply to any `substructure cloud ...` invocation:

- `--org <ID>` / `--app <ID>`: target a specific org or app, overriding `substructure.toml`.
- `--config <PATH>`: use a `substructure.toml` from a custom path.
- `--credentials <PATH>`: use a credentials file from a custom path.
- `--url <URL>`: override the cloud API URL (also reads `$SUBS_API_URL`).
- `--json`: emit machine-readable JSON instead of formatted tables.
- `--no-interaction` (`-n`): never prompt; fail if input would be required. Useful in scripts and CI.

