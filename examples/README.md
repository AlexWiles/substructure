# Examples

Each example is a self-contained project with its own dependencies and lockfile —
there is no shared workspace, so this folder can also hold examples in languages
other than TypeScript.

The TypeScript examples depend on the packages in this repo via `file:`, so build
them once from the repo root before running an example:

```sh
pnpm -r build            # builds @substructure.ai/sdk, /runtime, /cli
```

Then, in any example directory:

```sh
npm install
npm start                # or: npm run dev / npm run typecheck — see the example's package.json
```

## Language examples (no SDK)

The worker contract is one JSON request in, one JSON response out, so an agent
needs no SDK. `chat-javascript`, `chat-python`, `chat-go`, and `chat-elixir` each
implement the same basic chattable agent as a single HTTP handler in that
language. They don't use the packages in this repo (no `pnpm build` needed) —
each runs against a local server (`substructure serve --dev --worker-url …`); see
the example's README.
