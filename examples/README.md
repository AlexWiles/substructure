# Examples

Each example is a self-contained project with its own dependencies and lockfile —
there is no shared workspace, so this folder can also hold examples in languages
other than TypeScript.

The TypeScript examples depend on the packages in this repo via `link:`, so build
them once from the repo root before running an example:

```sh
pnpm -r build            # builds @substructure.ai/sdk, /runtime, /cli
```

Then, in any example directory:

```sh
pnpm install --ignore-workspace
pnpm start               # or: dev / typecheck — see the example's package.json
```

`--ignore-workspace` keeps the install scoped to the single example instead of
the repo's root workspace.
