# node-embedded

The smallest possible agent: no server, no worker, no client. The runtime
runs in-process and persists state + the event log to a local SQLite file
(`agent.db`). If the process crashes mid-turn, restarting and replaying the
log resumes where it left off.

Use this shape when you want a single Node binary with no infrastructure.

## Run

```sh
export OPENROUTER_API_KEY=sk-or-...
pnpm install
pnpm start
```
