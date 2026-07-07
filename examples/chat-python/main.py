# A complete chat agent. No SDK — one FastAPI POST handler.
# The engine POSTs a decision request; this returns the next actions.
#
# Point a local Substructure server at it:
#   substructure serve --dev --provider openrouter --worker-url http://localhost:4444
from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/")
async def decide(request: Request):
    req = await request.json()
    t = req["trigger"]
    decision = {}

    # The client sent the conversation → prompt the model.
    if t["type"] == "client.messages":
        decision = {"messages": t["messages"], "actions": [{
            "type": "llm.call", "handler": "server", "stream": True,
            "request": {"model": "anthropic/claude-sonnet-4-6", "messages": t["messages"]}}]}

    # The model answered → record it, end the turn.
    if t["type"] == "llm.finished":
        decision = {"messages": [*req["messages"], t["message"]],
                    "actions": [{"type": "done", "data": t["message"]["content"]}]}

    return {"actions": [], **decision}
