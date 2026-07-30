# node-hono-git-search

An agent with tools that explore the source of a single git repo, served with
[Hono](https://hono.dev).

At startup the worker shallow-clones the repo into a temp cache dir and exposes
three tools backed by git:

- `list_files` — `git ls-files`, optionally filtered by a glob.
- `search_code` — `git grep`, returning `path:line:text` matches.
- `read_file` — `git show HEAD:<path>`, returning a bounded line window.

All three cap their output so a broad query or huge file can't blow the model's
context — `read_file` reads at most 200 lines and pages via `start_line`. Reads
are confined to files tracked in the repo. No API, no token, no rate limit. Edit
the `REPO` constant in `server.mjs` to point it elsewhere; it uses the repo's
default branch.

Requires `git` on your PATH.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker** (the first run clones the repo):

```sh
npm install
node server.mjs
```

**2. Send a message with the CLI**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run -c substructure.toml \
    --input '{"type":"client.message","message":{"role":"user","content": "where is the tool.execute trigger handled?"}}'
```
