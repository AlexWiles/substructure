# Per-user MCP credentials

One connection, `credential = "user"`: each person connects their own
account, and a shared conversation can never use it. The mock server plays
an OAuth-protected mail service whose `whoami` tool answers with the
connected account.

```sh
node mock-server.mjs &

# Optional: seal stored credentials at rest. Without it they are plain
# text in the local database, and the engine warns at startup.
export SUBS_SECRET_KEY="$(openssl rand -hex 32)"
export ANTHROPIC_API_KEY=sk-ant-...

subs list                    # mail … not connected
subs auth mcp.mail              # browser consent; the mock auto-approves
subs run "Whose mail account is connected?"
```

The credential lands in the `local` person's slot — `connector_credentials`
holds a reference, `secrets` holds the value, sealed when the key is set.
