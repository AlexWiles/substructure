# A chat agent whose LLM calls run on the worker, served with FastAPI.
import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import OpenAI

client = OpenAI()


def sse(event, data):
    return f"event: {event}\ndata: {data}\n\n"


def decide(req):
    trigger = req["trigger"]

    # `llm = "byo"` in subs.toml is a `type = "worker"` block, so the
    # calls come back here as `llm.execute`, shaped by that block's `format`.
    # The request is already a Chat Completions body; raw stream chunks and
    # the final completion go straight back.
    if trigger["type"] == "llm.execute":

        def events():
            with client.chat.completions.stream(**trigger["request"]) as stream:
                for event in stream:
                    if event.type == "chunk":
                        yield sse("llm.token.delta", event.chunk.model_dump_json())
                actions = [
                    {
                        "type": "llm.result",
                        "response": stream.get_final_completion().model_dump(mode="json"),
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
