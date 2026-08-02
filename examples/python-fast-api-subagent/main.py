# A chat agent that delegates to a weather sub-agent, served with FastAPI.
from fastapi import FastAPI, Request


def assistant(req):
    if req["trigger"]["type"] == "session.start":
        return {
            "agent": {
                **req["proposed"]["agent"],
                "system": "Delegate weather questions to the weather agent.",
                "sub_agents": [
                    {
                        "id": "weather",
                        "description": "Answers weather questions for a city.",
                    }
                ],
            }
        }

    return req["proposed"]


def weather(req):
    trigger = req["trigger"]

    if trigger["type"] == "session.start":
        return {
            "agent": {
                **req["proposed"]["agent"],
                "system": "You report the weather. Look it up, then answer in one line.",
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "input": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            }
        }

    if trigger["type"] == "tool.execute":
        city = trigger["input"]["value"]["city"]
        return {
            "actions": [
                {
                    "type": "tool.result",
                    "result": {"city": city, "tempF": 68, "sky": "clear"},
                }
            ]
        }

    return req["proposed"]


# One worker, two agents told apart by `agent_id`.
def decide(req):
    if req["agent_id"] == "assistant":
        return assistant(req)
    if req["agent_id"] == "weather":
        return weather(req)


app = FastAPI()


@app.post("/")
async def worker(request: Request):
    return decide(await request.json())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=4444)
