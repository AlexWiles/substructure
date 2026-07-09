# A complete chat agent with a tool. No SDK — one FastAPI POST handler.
# The engine POSTs a decision request; this returns the next actions.
#
# The worker accepts every decision the engine has a default for (`proposed`)
# and authors only the two that are genuinely its own: the LLM request (the
# agent's identity) and the tool execution. Everything else — tool results,
# model replies, model failures, even broken or hallucinated tool calls —
# flows through the proposed-first line at the top.
#
# Point a local Substructure server at it:
#   subs serve --dev --provider anthropic --worker-url http://localhost:4444
from datetime import datetime, timezone

from fastapi import FastAPI, Request

TOOLS = [
    {
        "info": {
            "name": "get_current_time",
            "description": (
                "Get the current UTC date and time. "
                "Call this whenever the user asks what time or date it is."
            ),
        },
        "exec": lambda args: datetime.now(timezone.utc).isoformat(),
    }
]

app = FastAPI()


@app.post("/")
async def decide(request: Request):
    req = await request.json()

    # The engine proposes actions for a default agent tool loop
    if req["proposed"]:
        return req["proposed"]

    t = req["trigger"]

    # The client sent the conversation → record it, prompt the model.
    if t["type"] == "client.messages":
        return {
            "messages": t["messages"],
            "actions": [
                {
                    "type": "llm.call",
                    "stream": True,
                    "request": {
                        "model": "claude-haiku-4-5-20251001",
                        "messages": t["messages"],
                        "tools": [tool["info"] for tool in TOOLS],
                    },
                }
            ],
        }

    # A declared tool call with valid arguments → run it, answer.
    # Invalid tool calls will have a proposal from the engine already
    if t["type"] == "tool.execute":
        tool = next(tool for tool in TOOLS if tool["info"]["name"] == t["name"])
        return {"actions": [{"type": "tool.result", "result": tool["exec"](t["input"]["value"])}]}

    return {"actions": []}
