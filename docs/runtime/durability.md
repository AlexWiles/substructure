---
title: Durability
---

# Durability

Substructure persists every step of a run so crashes and restarts do not lose
work.

## Recovery model

If a process exits mid-run, the runtime replays from the last durable checkpoint.

## Guarantees

- State transitions are persisted
- Retries are coordinated by the runtime
- Long-running work survives process restarts
