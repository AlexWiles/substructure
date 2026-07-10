# A chat agent with a tool, served with FastAPI.
from datetime import datetime, timezone

from fastapi import FastAPI, Request

TOOLS = [
    {
        "name": "get_current_time",
        "description": "Get the current UTC date and time.",
        "exec": lambda: datetime.now(timezone.utc).isoformat(),
    },
    {
        "name": "get_current_time_zone",
        "description": "Get the user's current timezone",
        "exec": lambda: datetime.now().astimezone().tzname(),
    },
]


def decide(req):
    trigger = req["trigger"]

    if trigger["type"] == "session.start":
        # The engine will use this agent config to generate proposed actions.
        return {
            "agent": {
                "model": "claude-haiku-4-5-20251001",
                "stream": True,
                "tools": [
                    {"name": t["name"], "description": t["description"]} for t in TOOLS
                ],
            }
        }

    # Run our tool when the model calls it.
    if trigger["type"] == "tool.execute":
        tool = next(t for t in TOOLS if t["name"] == trigger["name"])
        return {"actions": [{"type": "tool.result", "result": tool["exec"]()}]}

    # Accept the engine's proposal for every other decision.
    return req["proposed"]


app = FastAPI()


@app.post("/")
async def worker(request: Request):
    return decide(await request.json())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=4444)
