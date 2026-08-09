---
title: Workers
group: Building agents
---

A worker is your code. It is an HTTP endpoint. The engine asks it before each
step of a turn, and your code answers.

Use a worker to change any decision the engine would make. Run a tool, rewrite a
prompt, swap the model, or pause for a person.

There is no SDK. The engine POSTs JSON and reads JSON back.

## The smallest worker

The engine proposes each step. Return the proposal to accept it.

```javascript title="server.mjs"
import { createServer } from "node:http";

function decide({ proposed }) {
    return proposed;
}

const server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(decide(JSON.parse(body))));
    });
});

server.listen(4444);
```

That is a complete agent. It behaves the same as an agent with no worker.

## Point an agent at it

```toml title="substructure.toml"
[agent.oncall]
llm = "openrouter"
model = "anthropic/claude-sonnet-4-5"
worker = "http://localhost:4444"
```

`worker` selects who decides. Only the agents that name a worker use one. The
rest stay with the engine, in the same project and the same file.

Run the worker in one terminal and a turn in another.

```sh
node server.mjs
subs run --agent oncall -o pretty "hi"
```

See [Local development](./160-local-development.md) for the local loop.

## Change one trigger at a time

Start from `return proposed` and handle the triggers you care about.

This worker declares a tool and runs it.

```javascript title="server.mjs"
const tools = [
    {
        name: "get_current_time",
        description: "Get the current UTC date and time.",
        exec: () => new Date().toISOString()
    }
];

function decide({ trigger, proposed }) {
    // Add our tools to the config from the file.
    if (trigger.type === "session.start") {
        return {
            agent: {
                ...proposed.agent,
                tools: tools.map(({ name, description }) => ({ name, description }))
            }
        };
    }

    // Run the tool when the model calls it.
    if (trigger.type === "tool.execute") {
        const tool = tools.find((t) => t.name === trigger.name);
        return { actions: [{ type: "tool.result", result: tool.exec() }] };
    }

    return proposed;
}
```

```sh
subs run --agent oncall -o pretty "what time is it?"
```

The model calls the tool. Your worker runs it. The engine records the result and
prompts the model again.

## Verify the signature

The engine signs every decision it sends. Check the signature before you act on
one.

```javascript title="server.mjs"
import { createHmac, timingSafeEqual } from "node:crypto";

const SECRET = process.env.SUBS_SIGNING_SECRET;

function verify(body, header) {
    const expected = "sha256=" + createHmac("sha256", SECRET).update(body).digest("hex");
    const a = Buffer.from(expected);
    const b = Buffer.from(header ?? "");
    return a.length === b.length && timingSafeEqual(a, b);
}
```

Refuse a request that does not match.

```javascript
if (!verify(body, req.headers["x-substructure-signature"])) {
    res.writeHead(401).end();
    return;
}
```

The cloud creates a signing secret for each agent that has a worker. Read it
with `subs agents secret <id>`. See [Authentication](./190-auth.md).

## Recipes

Each of these is a branch inside `decide`.

**Refuse a tool call.**

```javascript
if (trigger.type === "tool.execute" && trigger.name === "delete_account") {
    return { actions: [{ type: "tool.error", error: "Not allowed here." }] };
}
```

**Change the model mid-conversation.** Return an `agent` on any decision.

```javascript
if (trigger.type === "client.messages" && isLongTask(trigger.messages)) {
    return { ...proposed, agent: { ...agent, model: "anthropic/claude-opus-4-5" } };
}
```

**Give the model a different tool set later in the conversation.**

```javascript
return { ...proposed, agent: { ...agent, tools: toolsFor(state.mode) } };
```

**Remember something between turns.** Write `state` on the response. It comes
back on every later request. See [Agent state](./90-state.md).

```javascript
return { ...proposed, state: { turns: (state?.turns ?? 0) + 1 } };
```

**Ask a person.** Pause the branch. The turn stays open and uses no compute. See
[Interrupts](./100-interrupts.md).

```javascript
return { actions: [{ type: "interrupt", reason: "confirm", payload: { message: "Send it?" } }] };
```

**End the turn early.**

```javascript
return { actions: [{ type: "done" }] };
```

**Answer a tool later.** Return an empty decision and report the result when the
work finishes. See [Deferred tools](./110-deferred-tools.md).

```javascript
if (trigger.type === "tool.execute") {
    startRender(trigger.id, trigger.input.value);
    return {};
}
```

## One worker, several agents

Every request carries `agent_id`. Route on it.

```javascript
const decide = (req) => (req.agent_id === "poet" ? poet(req) : assistant(req));
```

## Next

- [Tool calls](./60-tools.md): schemas, errors, and where a tool runs.
- [Agent state](./90-state.md): memory that travels with the session.
- [Interrupts](./100-interrupts.md): pause for a person.
- [Protocol](./230-protocol.md): every field of the request and the response.
- [Typed bindings](./270-typed-bindings.md): generate the types for your language.
