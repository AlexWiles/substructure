# A complete chat agent served with FastAPI.
from fastapi import FastAPI, Request


def decide(req):
    trigger = req["trigger"]

    if trigger["type"] == "session.start":
        # The engine will use this agent config to generate proposed actions.
        return {
            "agent": {
                "model": "claude-haiku-4-5-20251001",
                "stream": True,
            }
        }

    # Accept the engine's proposal for every other decision.
    return req["proposed"]


app = FastAPI()


@app.post("/")
async def worker(request: Request):
    return decide(await request.json())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=4444)
