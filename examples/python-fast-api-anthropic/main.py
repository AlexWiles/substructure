# A chat agent whose LLM calls run on the worker, served with FastAPI.
import json

from anthropic import Anthropic
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

client = Anthropic()


def sse(event, data):
    return f"event: {event}\ndata: {data}\n\n"


def decide(req):
    trigger = req["trigger"]

    # `llm = "byo"` in substructure.toml is a `type = "worker"` block, so the
    # calls come back here as `llm.execute`, shaped by that block's `format`.
    # The request is already a Messages API body; raw stream events and the
    # final message go straight back.
    if trigger["type"] == "llm.execute":

        def events():
            with client.messages.stream(**trigger["request"]) as stream:
                for event in stream:
                    yield sse("llm.token.delta", event.model_dump_json())
                actions = [
                    {
                        "type": "llm.result",
                        "response": stream.get_final_message().model_dump(mode="json"),
                    }
                ]
                yield sse("decision.result", json.dumps({"actions": actions}))

        return StreamingResponse(events(), media_type="text/event-stream")

    # Accept the engine's proposal for every other decision.
    return req["proposed"]


app = FastAPI()


@app.post("/")
async def worker(request: Request):
    return decide(await request.json())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=4444)
