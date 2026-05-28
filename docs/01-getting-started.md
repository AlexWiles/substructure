---
title: Getting Started
---

Substructure is an open-source engine for building durable, long-running AI agents using just an HTTP endpoint hosted on your infrastructure, in your code.

Substructure drives the agentic loop, handling retries, sub-agent supervision, llm calls, real-time event streaming and more. Tool execution and agent decisions live in your codebase on your infrastructure.

The cloud hosted version is the simplest way to get started developing and go live to production.

Components:

- **Cloud** — The hosted Substructure engine at https://app.substructure.ai. Sign up, point the SDK at it, and run production agent sessions without operating any infrastructure or worrying about LLM providers. This is the default way to use Substructure.
- **CLI** — `@substructure.ai/cli`. Provisions projects, observes live sessions, and debugs runs against the cloud from the terminal. Built to pair with coding agents.
- **SDK** — First-party TypeScript SDK with a middleware DSL for state, message history, tools, sub-agents, and the LLM loop. Workers in other languages can implement the wire protocol directly.
- **Local** — Self-hosted Substructure server (Rust, SQLite). Escape hatch for development and air-gapped deploys; start with `substructure start`.
- **Embedded** — The same server in-process via NAPI (`sub.embedded({...})`). Useful for tests and single-binary deploys.

## Why substructure.ai

Substructure is a durable execution runtime for AI agents. You write a stateless HTTP handler that takes a trigger and returns actions. The cloud handles everything else.
What the cloud handles: durability (event log, retries, crash recovery, idempotency), multi-tenant session state, streaming to clients, and auth (short-lived JWTs for browsers, API keys for service-to-service).
What stays in your code: the HTTP handler, the prompts, the control flow, and a few agent.use(...) middleware calls.
Properties:
- Robust by default. Crashes, retries, duplicate requests, partial tool failures, and reconnecting clients are absorbed by the event log and idempotent replay.
- Scales without rearchitecting. The day-one agent keeps working as sessions, tool calls, and sub-agents grow.
- Portable. Workers are plain HTTP; state is opaque bytes. The same agent runs self-hosted or embedded.

Use Substructure when an app needs durable, observable, multi-tenant agent behavior and you'd rather call a hosted engine than operate workflow infrastructure — while keeping prompts and control flow in your own codebase.

