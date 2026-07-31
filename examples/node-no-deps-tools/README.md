# node-no-deps-tools

A chattable agent with two tools, served with Node's built-in `http` server. No dependencies.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker**:

```sh
node server.mjs
```

**2. Send a message with the CLI**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run -c substructure.toml --agent my-agent "what time is it in my timezone?"
```
