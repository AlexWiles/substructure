---
title: Overview
---

Substructure is an open-source engine for building durable, long-running AI agents using just an HTTP endpoint hosted on your infrastructure, in your code.

Substructure drives the agentic loop, handling retries, sub-agent supervision, llm calls, real-time event streaming and more. Tool execution and agent decisions live in your codebase on your infrastructure.

The cloud hosted version is the simplest way to get started developing and go live to production.

Components:

- **Cloud** The hosted Substructure engine at https://app.substructure.ai. Sign up, point the SDK at it, and run production agent sessions without operating any infrastructure or worrying about LLM providers. This is the default way to use Substructure.
- **CLI** `@substructure.ai/cli`. Provisions projects, observes live sessions, and debugs runs against the cloud from the terminal. Built to pair with coding agents.
- **SDK** First-party TypeScript SDK with a middleware DSL for state, message history, tools, sub-agents, and the LLM loop. Workers in other languages can implement the wire protocol directly.
- **Local** Self-hosted Substructure server (Rust, SQLite). Escape hatch for development and air-gapped deploys; start with `substructure start`.
- **Embedded** The same server in-process via NAPI (`sub.embedded({...})`). Useful for tests and single-binary deploys.

## Why substructure.ai

- Substructure handles the nasty stuff involved in building production ready agents that can integrate into larger software systems.
- Portable. Workers are stateless functions. Runnable as stateless HTTP endpoints on the cloud/self-hosted, or embedded directly in-process.

