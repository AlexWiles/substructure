# A complete chat agent served with FastAPI.
from fastapi import FastAPI, Request


# The whole worker: accept the engine's proposal for every decision. The agent
# is declared in substructure.toml, and arrives as the `session.start`
# proposal, so there is nothing to author until you want to override something.
def decide(req):
    return req["proposed"]


app = FastAPI()


@app.post("/")
async def worker(request: Request):
    return decide(await request.json())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=4444)
