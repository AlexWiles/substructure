# sub-agent

Agent-to-agent delegation. A parent assistant exposes a child weather
agent as a tool. The child runs its own decision loop (its own system
prompt, tools, state) and returns a result back to the parent.

Use this shape when you want to compose specialists rather than stuff
every tool into a single agent. Failures isolate to the child; token
and cost usage rolls up to the parent.

## Run

```sh
export OPENROUTER_API_KEY=sk-or-...
npm install
npm start
```
