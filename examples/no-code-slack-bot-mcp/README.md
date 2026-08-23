# no-code-slack-bot-mcp

A Slack bot with tools, in one file. There is no worker and no code.

The tools come from an [MCP](https://modelcontextprotocol.io) server. The engine
holds the connection and runs each call, so nothing about it reaches Slack. Each
call the bot makes is a task card in the thread.

This example reads public GitHub repositories through
[DeepWiki](https://deepwiki.com), which needs no credential, so it runs as it is.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Deploy the file and give it a key:

```sh
subs login
subs apply
subs auth llm.openrouter
```

Install the bot into your workspace:

```sh
subs slack connect
```

Mention the bot and ask about a repository.

## The connection

The agent names a connection. The connection names a URL:

```toml
[agent.docs]
mcp = ["deepwiki"]

[mcp.deepwiki]
url = "https://mcp.deepwiki.com/mcp"
```

The tools belong to the agent that answers. They are not a request the model can
refuse. A server that needs a credential takes one more step,
`subs auth <path>`, and the credential goes to the deployment. See
[no-code-mcp](../no-code-mcp).
