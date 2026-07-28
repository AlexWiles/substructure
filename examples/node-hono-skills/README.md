# node-hono-skills

An agent that loads [Agent Skills](https://agentskills.io) — folders of `SKILL.md`
instructions — only when a task needs them, served with [Hono](https://hono.dev).

Each skill is a self-contained folder. Its `name` + `description` sit in the
system prompt; the full instructions and any tools load on demand.

```
skills/
├── commit-messages/
│   └── SKILL.md      # instructions only
└── unit-conversion/
    ├── SKILL.md      # instructions
    └── tools.mjs     # the function they call
```

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker**:

```sh
npm install
node server.mjs
```

**2. Drive a session with the CLI.** The model picks a skill from the catalog,
loads it, then acts:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run \
    --worker-url http://localhost:4444 \
    --agent skills \
    --llm-provider anthropic \
    --output pretty \
    --session skills-demo \
    --input '{"type":"client.message","message":{"role":"user","content":"how far is 42 km in miles?"}}'
```

Ask for a commit message instead to load a skill that ships no tools:

```sh
    --input '{"type":"client.message","message":{"role":"user","content":"commit message for adding rate limiting to the login route"}}'
```

## Add a skill

Drop a `skills/<name>/SKILL.md` with `name` + `description` frontmatter. For
tools, add a `tools.mjs` that default-exports `[{ name, description, input, exec }]`.
