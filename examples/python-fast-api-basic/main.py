# A complete chat agent. No SDK — one FastAPI POST handler.
# The engine POSTs a decision request; this returns the next actions.
#
# The worker accepts every decision the engine has a default for (`proposed`)
# and authors only the one that is genuinely its own: the LLM request — the
# agent's identity. Everything else — model replies and model failures — flows
# through the proposed-first line at the top.
#
# Point a local Substructure server at it:
#   subs serve --dev --provider anthropic --worker-url http://localhost:4444
from fastapi import FastAPI, Request

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
                    },
                }
            ],
        }

    return {"actions": []}
