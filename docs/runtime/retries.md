---
title: Retries
---

# Retries

Retries are managed by the server so worker code stays focused on agent logic.

## Retry policy

You can configure timeout and max retries in middleware.

## Failure handling

Transient failures are retried automatically without re-running successful steps.
