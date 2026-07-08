# A complete chat agent. No SDK — one FastAPI POST handler.
# The engine POSTs a decision request; this returns the next actions.
#
# Point a local Substructure server at it:
#   substructure serve --dev --provider anthropic --worker-url http://localhost:4444
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
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        "exec": lambda args: datetime.now(timezone.utc).isoformat(),
    }
]


def llm_call_action(messages):
    {
        "type": "llm.call",
        "handler": "server",
        "stream": True,
        "request": {
            "model": "claude-haiku-4-5-20251001",
            "messages": messages,
            "tools": [t["info"] for t in TOOLS],
        },
    }


app = FastAPI()


@app.post("/")
async def decide(request: Request):
    req = await request.json()
    t = req["trigger"]

    # The client sent the conversation → prompt the model.
    if t["type"] == "client.messages":
        return {"messages": messages, "actions": [llm_call_action(t["messages"])]}

    if t["type"] == "tool.execute":
        tool = next(tool for tool in TOOLS where tool["name"] ==  t["name"], None)
        if tool is None:
            return {"type": "tool.result", "id": t["id"], "result": "Tool not found"}


    # The model answered.
    if t["type"] == "llm.finished":
        new_message = t["message"]

        tool_use_actions = [
            {
                "type": "tool.call",
                "name": c["function"]["name"],
                "arguments": c["function"]["arguments"],
            }
            for c in new_message["content"]
            if c["type"] == "tool_use"
        ]

        # append new message to the requests message array
        # we return it to update the message list in the engine
        messages = [*req["messages"], new_message]

        if tool_use_actions:
            return {
                "messages": messages,
                "actions": [llm_call_action(messages)],
            }
        else:
            return {
                # mark the turn done
                "messages": messages,
                "actions": [{"type": "done", "data": message["content"]}],
            }

    # return no actions
    return {"actions": []}
