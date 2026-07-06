# frontend-tool

A browser chat UI where the agent's tools run *in the browser*. The agent
has two tools:

- `get_user_location` — reads `navigator.geolocation`
- `set_theme(background, accent)` — mutates the page's CSS variables

Both are impossible from a backend worker. The tools are declared with
`handler: "client"` so the engine never dispatches `tool.execute` to the
worker — it just emits `tool.call.requested` on the session stream. The
browser sees that event, executes the tool locally, and posts the result
via `settleEffect`. The agent resumes as if the tool had returned
synchronously.

Auth is the standard browser pattern: the Hono server mints a
short-lived JWT for the page via `backend.mintClientToken(...)`, and the
page uses `sub.frontend.client({ token })` to talk to the Substructure
server directly.

## Run

In one terminal, start a local Substructure server pointed at this worker:

```sh
export OPENROUTER_API_KEY=sk-or-...
substructure start --dev --port 9000 --worker-url http://localhost:3333/agent
```

In another terminal, start the Hono worker + static server:

```sh
npm install
npm start
```

Open <http://localhost:3333> and try:

- *"where am I?"* — your browser will prompt for geolocation permission.
- *"make this page look like the ocean"* — the agent picks colors and
  applies them.
