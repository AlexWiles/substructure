---
name: respond
description: Respond to an on-call incident from the team's runbooks. Use when something is down, slow, or full.
---

Answer from the runbooks, not from memory.

1. `list_dir` to see which runbooks exist.
2. `read_file` the one that matches the incident. Read it whole.
3. Give the steps in the runbook's own order, with its exact commands.
4. End with whether to escalate, per references/escalation.md.

No runbook matches: say so and escalate. Do not improvise steps.
