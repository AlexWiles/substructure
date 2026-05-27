---
title: Runtime Overview
---

# Runtime overview

Substructure separates agent systems into three parts:

- **Workers** for your agent logic
- **Server** for orchestration and durability
- **Clients** for user-facing and backend integrations

This separation keeps your code portable while making execution resilient.

## Durable execution

Every decision is persisted so runs can resume safely after crashes and
restarts.

## Middleware model

Agent capabilities are composed through middleware, so you can add and
replace behavior intentionally.
