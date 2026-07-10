# A chat agent that verifies each request's signature before deciding.
import hashlib
import hmac

from fastapi import Depends, FastAPI, HTTPException, Request


def decide(req):
    trigger = req["trigger"]

    if trigger["type"] == "session.start":
        return {"agent": {"model": "claude-haiku-4-5-20251001", "stream": True}}

    # Accept the engine's proposal for every other decision.
    return req["proposed"]


app = FastAPI()

# Hardcoded for the example only. In production load this from a secret store
# and pass the same value to the CLI with --signing-secret.
SIGNING_SECRET = "dev-secret-not-for-production"


# Reject any request whose signature doesn't check out. The engine signs each
# request with HMAC-SHA256 over the raw body, sending a `sha256=<hex>` signature.
async def verify_signature(request: Request):
    body = await request.body()
    expected = (
        "sha256=" + hmac.new(SIGNING_SECRET.encode(), body, hashlib.sha256).hexdigest()
    )
    received = request.headers.get("X-Substructure-Signature", "")

    # Constant-time compare so a mismatch leaks no timing.
    if not hmac.compare_digest(received, expected):
        raise HTTPException(status_code=401, detail="invalid signature")


@app.post("/", dependencies=[Depends(verify_signature)])
async def worker(request: Request):
    return decide(await request.json())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=4444)
