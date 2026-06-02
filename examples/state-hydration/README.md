# state-hydration

The persisted (wire) state holds only compact **references** — a history id and
a todos id. A hydration middleware loads the referenced data into rich objects
the agent works with, then on the way out saves them back and restores the
references, so only the ids ever ride the wire.

```
wire / persisted:   { historyId, todosId }                    ← tiny
       ↓ hydrate (load from DB)
runtime / agent:    { historyId, todosId, messages, todos }   ← what tools & middleware see
       ↑ dehydrate (save to DB, drop the heavy fields)
wire / persisted:   { historyId, todosId }
```

## Transform, not contribution

A state slice **adds** keys to the state type (`S -> S & A`). This is different:
the hydration middleware **swaps the whole shape** (`Refs -> Hydrated`) for
everything downstream. It's a plain `MiddlewareFn<In, Out>`:

```ts
const hydrate: MiddlewareFn<Refs, Hydrated> = async (ctx, next) => {
    const hydrated = { ...load from DB... };
    const res = await next({ ...ctx, state: hydrated });   // downstream now sees Hydrated
    ...save to DB...
    return { ...res, state: refsOnly };                    // wire keeps only the ids
};
```

`.use(hydrate)` uses the builder's transform overload
(`use<Out>(mw: MiddlewareFn<S, Out>): HandlerBuilder<Out>`), which **replaces**
the state type with `Out`. So every middleware and tool below `hydrate` sees the
fully-typed `Hydrated` shape — `state.messages: Message[]`, `state.todos: Todo[]`.

## Two things the types won't enforce

- **`dehydrate` is mandatory.** `jsonState` encodes whatever state comes back. If
  you don't restore the references, the hydrated objects get serialized onto the
  wire and balloon every turn. Here the way-out half saves to the DB and returns
  just `{ historyId, todosId }`.
- **Ordering matters.** `hydrate` must sit after the slice that establishes
  `Refs` and before anything meant to see `Hydrated`.

## Ids as keys

`todosId` defaults to the user id (todos persist across all of a user's
sessions) and `historyId` to the session id (conversation is per session). The
ids live in state, so you can repoint either one without touching anything else.

## Run

```sh
export OPENROUTER_API_KEY=sk-or-...
pnpm install
pnpm start
```

Inspect what ended up where:

```sh
cat hydration-db/todos-*.json     # the actual todo items, per user
cat hydration-db/history-*.json   # the conversation, per session
```

The agent's wire-state blob (inside the engine's `agent.db`) only ever contains
`{ historyId, todosId }`. The messages and todos never go through Substructure.
