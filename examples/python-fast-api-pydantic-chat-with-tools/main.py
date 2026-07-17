# A chat agent with tools, served with FastAPI. The Pydantic models are
# generated from schemas/protocol.schema.json (see README), so FastAPI
# validates every decision request at the boundary.
from datetime import datetime, timezone

from fastapi import FastAPI

from protocol import AgentConfig, AgentTool, DecisionRequest, DecisionResponse

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


def decide(req: DecisionRequest) -> DecisionResponse:
    trigger = req.trigger

    if trigger.type == "session.start":
        # The engine will use this agent config to generate proposed actions.
        return DecisionResponse(
            agent=AgentConfig(
                model="claude-haiku-4-5-20251001",
                stream=True,
                tools=[
                    AgentTool(name=t["name"], description=t["description"])
                    for t in TOOLS
                ],
            )
        )

    # Run our tool when the model calls it.
    if trigger.type == "tool.execute":
        tool = next(t for t in TOOLS if t["name"] == trigger.name)
        return DecisionResponse(
            actions=[{"type": "tool.result", "result": tool["exec"]()}]
        )

    # Accept the engine's proposal for every other decision.
    return req.proposed


app = FastAPI()


@app.post("/")
async def worker(req: DecisionRequest):
    decision = decide(req)
    return decision.model_dump(exclude_none=True, mode="json") if decision else None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=4444)
