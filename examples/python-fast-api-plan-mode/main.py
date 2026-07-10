# Plan mode: build a plan over several turns, then flip a switch to execute it.
# Uses session state to store the plan across turns.
from fastapi import FastAPI, Request

PLANNING_TOOLS = [
    {
        "name": "add_step",
        "description": "Append a step to the plan.",
        "input": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "exec": lambda plan, args: plan.append({"text": args["text"], "done": False}),
    }
]

EXECUTING_TOOLS = [
    {
        "name": "complete_step",
        "description": "Mark the numbered plan step done.",
        "input": {
            "type": "object",
            "properties": {"number": {"type": "integer"}},
            "required": ["number"],
        },
        "exec": lambda plan, args: plan[args["number"] - 1].update(done=True),
    }
]

# One profile per mode: planning fills the checklist, executing walks it.
PROFILES = {
    "planning": {
        "model": "claude-haiku-4-5-20251001",
        "stream": True,
        "system": "You are planning. Break the goal into steps; call add_step for each. Do not execute yet.",
        "tools": PLANNING_TOOLS,
    },
    "executing": {
        "model": "claude-haiku-4-5-20251001",
        "stream": True,
        "system": "You are executing. Work through each pending step in order; call complete_step with its number when done.",
        "tools": EXECUTING_TOOLS,
    },
}


# The agent config the engine reads: the profile with each tool's exec stripped.
def agent(mode):
    profile = PROFILES[mode]
    return {
        "model": profile["model"],
        "stream": profile["stream"],
        "system": profile["system"],
        "tools": [
            {"name": t["name"], "description": t["description"], "input": t["input"]}
            for t in profile["tools"]
        ],
    }


def render_plan(plan):
    if not plan:
        return "(empty plan)"
    return "\n".join(
        f"{i + 1}. [{'x' if s['done'] else ' '}] {s['text']}"
        for i, s in enumerate(plan)
    )


def decide(req):
    trigger = req["trigger"]
    state = req.get("state") or {"mode": "planning", "plan": []}

    if trigger["type"] == "session.start":
        return {"agent": agent(state["mode"]), "state": state}

    # Flip modes with an action, not a message. Entering execution forks a fresh
    # branch seeded with the plan so the executor starts clean.
    if trigger["type"] == "client.action" and trigger["name"] == "set_mode":
        state["mode"] = (trigger.get("args") or {}).get("mode")

        if state["mode"] == "executing":
            seed = {"role": "user", "content": render_plan(state["plan"])}
            return {
                "agent": agent("executing"),
                "state": state,
                "messages": [seed],
                "actions": [{"type": "llm.call"}],
            }
        if state["mode"] == "planning":
            return {"agent": agent("planning"), "state": state}
        return {"state": state}

    if trigger["type"] == "tool.execute":
        tool = next(
            t for t in PROFILES[state["mode"]]["tools"] if t["name"] == trigger["name"]
        )
        # tools mutate the state
        tool["exec"](state["plan"], trigger["input"]["value"])
        # return updated state and the tool result
        return {
            "state": state,
            "actions": [{"type": "tool.result", "result": render_plan(state["plan"])}],
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
