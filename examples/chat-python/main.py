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

    # The model answered.
    if t["type"] == "llm.finished":
        new_message = t["message"]

        # extract any tool calls, turning them into actions
        tool_use_actions = [
            {
                "type": "tool.call",
                "name": c["function"]["name"],
                "arguments": c["function"]["arguments"],
            }
            for c in new_message["content"]
            if c["type"] == "tool_use"
        ]

        # append the new message to the message history
        messages = [*req["messages"], new_message]

        if tool_use_actions:
            # if there are tool calls, return the updated message list
            # and trigger tool calls by returning the options
            return {
                "messages": messages,
                "actions": [llm_call_action(messages)],
            }
        else:
            # if there are no tool calls, return the updated message list
            # and return the done action, so the turn ends
            return {
                "messages": messages,
                "actions": [{"type": "done", "data": message["content"]}],
            }


    if t["type"] == "tool.execute":
        tool = next([tool for tool in TOOLS if tool["name"] == t["name"]], None)
        if tool is None:
            return {"actions": [{"type": "tool.result", "result": "Tool not found"}]}
        else:
            result = tool["exec"](t["args"])
            return {"actions": [{"type": "tool.result", "result": result}]}
    if t["type"] == "tool.finished":
        new_message = {
            "role": "tool",
            "content": t["result"],
            "tool_call_id": t["id"],
            "name": t["name"],
        }
        messages = [*req["messages"], new_message]
        if req["pending_calls"] > 0:
            # if there are pending calls, we only update the message history
            return {"actions": [], "messages": messages}
        else:
            # if there are no pending calls, we update the message history and call the LLM
            return {"actions": [llm_call_action(messages)], "messages": messages}

    # return no actions
    return {"actions": []}
