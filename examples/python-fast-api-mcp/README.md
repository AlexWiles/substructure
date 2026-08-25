# python-fast-api-mcp

An agent whose tools come from an [MCP](https://modelcontextprotocol.io) server,
served with [FastAPI](https://fastapi.tiangolo.com).

The worker connects to an MCP server over stdio, exposes its tools to the model,
and forwards each tool call back to the MCP server. Here it runs the filesystem
server scoped to this directory; swap the command for any MCP server.

## Run

Install the CLI:

```sh
npm i -g @substructure.ai/cli
```

Two terminals.

**1. Start the worker**:

```sh
pip install -r requirements.txt
python3 main.py
```

**2. Send a message with the CLI**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
subs run -c subs.toml my-agent "what files are in this directory?"
```
