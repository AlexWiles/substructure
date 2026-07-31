# A chat agent whose tools come from an MCP server, served with FastAPI.
import os
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

mcp = {}


@asynccontextmanager
async def lifespan(app):
    # Connect to an MCP server over stdio and pull in its tools. Here we run the
    # filesystem server scoped to this directory, but any MCP server works.
    params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()],
    )
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        mcp["session"] = session
        mcp["tools"] = (await session.list_tools()).tools
        mcp["names"] = {t.name for t in mcp["tools"]}
        yield


async def decide(req):
    trigger = req["trigger"]

    if trigger["type"] == "session.start":
        # Offer every MCP tool to the model as-is.
        return {
            # The declared config arrives as the proposal; spread it to keep
            # the `llm` and `model` substructure.toml names.
            "agent": {
                **req["proposed"]["agent"],
                "stream": True,
                "tools": [
                    {"name": t.name, "description": t.description, "input": t.inputSchema}
                    for t in mcp["tools"]
                ],
            }
        }

    # Forward the call to the MCP server and settle with its output.
    if trigger["type"] == "tool.execute" and trigger["name"] in mcp["names"]:
        res = await mcp["session"].call_tool(
            trigger["name"], (trigger.get("input") or {}).get("value") or {}
        )
        text = "\n".join(c.text for c in res.content if c.type == "text")
        if res.isError:
            return {"actions": [{"type": "tool.error", "error": text}]}
        return {"actions": [{"type": "tool.result", "result": text}]}

    # Accept the engine's proposal for every other decision.
    return req["proposed"]


app = FastAPI(lifespan=lifespan)


@app.post("/")
async def worker(request: Request):
    return await decide(await request.json())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=4444)
