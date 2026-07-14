---
title: Quick start
group: Getting started
---

Build a chat agent, talk to it, give it a tool, pick the conversation back
up, then add a sub-agent. Two terminals, no dependencies.

## 1. Create the worker

Create a file called `server.mjs`:

```javascript title="server.mjs"
// A complete chat agent served with Node's built-in http server.
import { createServer } from "node:http";

function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        // The engine will use this agent config to generate proposed actions.
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true
            }
        };
    }

    // Accept the engine's proposal for every other decision.
    return proposed;
}

const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
        const decision = decide(JSON.parse(body));
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(decision ?? null));
    });
});

server.listen(4444, () =>
    console.log("worker listening on http://localhost:4444"));
```

Start it:

```sh
node server.mjs
```

## 2. Send a message

In another terminal, install the CLI:

```sh
npm install -g @substructure.ai/cli
```

Send a message:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --provider anthropic \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "hi"}}'
```

The reply streams to your terminal. The engine ran the loop; your worker made every decision.

## 3. Add a tool

Replace `decide` in `server.mjs`:

```javascript title="server.mjs"
const tools = [
    {
        name: "get_current_time",
        description: "Get the current UTC date and time.",
        exec: () => new Date().toISOString()
    }
];

function decide({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true,
                tools: tools.map(({ name, description }) => ({ name, description }))
            }
        };
    }

    // Run our tool when the model calls it.
    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        return { actions: [{ type: "tool.result", result: tool.exec() }] };
    }

    // Accept the engine's proposal for every other decision.
    return proposed;
}
```

Restart the worker and ask a question that needs the tool:

```sh
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --provider anthropic \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "what time is it?"}}'
```

The model calls the tool, your worker runs it, and the engine folds the result
back into the conversation.

## 4. Continue the conversation

Each run so far started a new session. After every run, the CLI prints the
command that continues it:

```
continue this session with:
  subs run --session <session-id> --worker-url http://localhost:4444 ...
```

Run it with a follow-up question:

```sh
subs run \
    --session <session-id> \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --provider anthropic \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "what was my first question?"}}'
```

The agent remembers. The whole session is persisted in `data.db`: messages,
tool calls, results. Stop everything, come back tomorrow, and it still
resumes.

## 5. Add a sub-agent

One worker can host several agents, told apart by `agent_id`. Replace the
agent code in `server.mjs` (everything above `createServer`):

```javascript title="server.mjs"
const tools = [
    {
        name: "get_current_time",
        description: "Get the current UTC date and time.",
        exec: () => new Date().toISOString()
    }
];

function assistant({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true,
                tools: tools.map(({ name, description }) => ({ name, description })),
                sub_agents: [{ id: "poet", description: "Writes a haiku on any topic." }]
            }
        };
    }

    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        return { actions: [{ type: "tool.result", result: tool.exec() }] };
    }

    return proposed;
}

function poet({ trigger, proposed }) {
    if (trigger.type === "session.start") {
        return {
            agent: {
                model: "claude-haiku-4-5-20251001",
                stream: true,
                system: "You are a poet. Respond with a single haiku."
            }
        };
    }

    return proposed;
}

// One worker, two agents told apart by agent_id.
function decide(req) {
    return req.agent_id === "poet" ? poet(req) : assistant(req);
}
```

The sub-agent appears to the model as a tool. Restart the worker and start a
conversation that uses both:

```sh
subs run \
    --worker-url http://localhost:4444 \
    --agent my-agent \
    --provider anthropic \
    --output pretty \
    --input '{"type":"client.message","message":{"role":"user","content": "what time is it? have the poet write a haiku about it"}}'
```

The assistant calls your tool, delegates to the poet, and folds the haiku into
its answer. The poet ran in its own session, and its cost and token usage
rolled up to the parent.

## Next

- [Core concepts](./20-concepts.md): sessions, turns, triggers, decisions.
- [Tool calls](./30-tools.md): schemas, validation, and where tools run.
- [Sub-agents](./80-sub-agents.md): delegation, child sessions, cost roll-up.
- [Durability](./110-durability.md): what's saved, and how runs recover.
- More agents in [`examples/`](../examples), in Node, Python, and Go.
