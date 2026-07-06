# state-hydration

The agent's state — a todo list — rides the decision envelope as `state`,
round-tripped every turn. The engine persists it between decisions and hands it
back next time; the agent reads `req.state`, the tools mutate it, and the loop
returns the new value.

There's no built-in tool state, so the tools are built per decision, each closing
over the live list. The agent is a plain custom `decide` function that builds a
`toolLoop` over those tools and hands it the state — `loop({ ...req, state })` —
which the loop echoes back out so the engine persists it.

```
wire / state:         { todos: Todo[] }          ← engine persists & round-trips it
       ↓ req.state
decide:               builds tools over state, loop runs them, todos mutate
       ↑ returns { ..., state }
wire / state:         { todos: Todo[] }           ← updated, handed back next turn
```

## The default loop, over state that rides the wire

`toolLoop(config)` returns the exact decision function the engine runs — prompt
assembly, `llm.call`, the tool round-trip, round gating. Nothing here is
overridden: the `decide` reads `req.state`, builds the loop, and calls it.

```ts
const todoAgent = agent<State>({
    name: "todo",
    decide: async (req) => {
        const state: State = { todos: req.state?.todos ?? [] };
        const loop = toolLoop<State>({
            llm: { model: "anthropic/claude-sonnet-4-6" },
            instructions: "Concise todo assistant. Use the tools to manage the list.",
            tools: todoTools(state),
        });
        return loop({ ...req, state });
    },
});
```

The `state` you pass into `loop({ ...req, state })` is echoed straight back out in
the decision the loop returns, so the engine persists it on the envelope and
hands it back as `req.state` on the next decision.

## Tools close over the state

There's no SDK-held tool state, so `todoTools(state)` builds the `tool()` defs
fresh each decision, with each `execute` closing over the live `state.todos`:

```ts
function todoTools(state: State) {
    return [
        tool({ name: "add_todo",   /* … */ execute: (args) => { state.todos.push(/* … */); /* … */ } }),
        tool({ name: "list_todos", /* … */ execute: () => state.todos.map(formatTodo).join("\n") }),
    ];
}
```

`toolLoop` turns them into the model's tool schemas and runs them on
`tool.execute`. Their edits land in `state.todos` — the same object the loop
echoes back out — so the todo list rides the wire with no manual plumbing.

## Run

```sh
export OPENROUTER_API_KEY=sk-or-...
npm install
npm start
```

The example runs a single turn against an in-memory engine (`db: ":memory:"`):
the user asks to add two todos and list them, and the stream prints each
user/assistant message, tool call, and tool result. Because the state rides the
wire, the same todo list would be handed straight back on the
next turn of a persisted session — the todos never live anywhere but the state
the agent returns.
