# Plan mode: build a plan over several turns, then flip a switch to execute it.
# No SDK — one FastAPI POST handler. The worker reads its `mode` from state and
# picks the model, prompt, and tool per mode: a planning tool builds a checklist,
# an executing tool walks it. The client flips modes with a `client.action`
# (`set_mode`), not a chat message, so the switch never lands in the transcript.
# Entering execution forks a fresh branch seeded with only the plan, so the
# executor starts clean.
#
# Point a local Substructure server at it:
#   subs serve --dev --provider anthropic --worker-url http://localhost:4444
from fastapi import FastAPI, Request

# ── Tools, per mode ───────────────────────────────────────────────────────────
# One tool per mode is the whole point: planning builds the checklist, executing
# walks it. Each lambda mutates the plan (a list of steps) in state, returns it.

PLANNING_TOOLS = [
    {
        "info": {
            "name": "add_step",
            "description": "Append a step to the plan.",
            "input": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
        "exec": lambda plan, args: plan.append({"text": args["text"], "done": False}) or plan,
    },
]

EXECUTING_TOOLS = [
    {
        "info": {
            "name": "complete_step",
            "description": "Mark the numbered plan step done (1-based, as shown in the plan).",
            "input": {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]},
        },
        "exec": lambda plan, args: plan[args["number"] - 1].update(done=True) or plan,
    },
]

TOOLS_BY_MODE = {"planning": PLANNING_TOOLS, "executing": EXECUTING_TOOLS}
ALL_TOOLS = {tool["info"]["name"]: tool for tools in TOOLS_BY_MODE.values() for tool in tools}

# ── Prompts and models, per mode ──────────────────────────────────────────────

PROFILES = {
    "planning": {
        "model": "claude-opus-4-8",
        "instructions": "\n".join(
            [
                "You are in PLANNING MODE.",
                "Work with the user to break the goal down into a checklist of concrete steps.",
                "Call add_step for each one. Do not execute anything yet.",
                "Be concise; after adding steps, summarize the plan in one line.",
            ]
        ),
    },
    "executing": {
        "model": "claude-haiku-4-5-20251001",
        "instructions": "\n".join(
            [
                "You are in EXECUTING MODE.",
                "Work through every pending step in the plan above, in order.",
                "For each one, do the work, then call complete_step with its number.",
                "Stop when every step is done.",
            ]
        ),
    },
}


def render_plan(plan):
    if not plan:
        return "(empty plan)"
    return "\n".join(f"  {i + 1}. [{'x' if s['done'] else ' '}] {s['text']}" for i, s in enumerate(plan))


def prompt(messages, mode):
    profile = PROFILES[mode]
    system = {"role": "system", "content": profile["instructions"]}
    return {
        "type": "llm.call",
        "stream": True,
        "request": {
            "model": profile["model"],
            "messages": [system, *messages],
            "tools": [tool["info"] for tool in TOOLS_BY_MODE[mode]],
        },
    }


app = FastAPI()


@app.post("/")
async def decide(request: Request):
    req = await request.json()
    state = req.get("state") or {"mode": "planning", "plan": []}
    t = req["trigger"]

    # The engine proposes actions for every default continuation (llm/tool
    # finishes, broken tool calls). Its re-issued prompt preserves the mode's
    # model, system message, and tools, so execution just rides the default loop.
    if req.get("proposed"):
        return req["proposed"]

    # A `set_mode` signal switches modes without touching the transcript. Entering
    # execution forks a fresh branch (a plan-only message with no id forks at the
    # root) and carries the plan over by submitting state on the forking decision.
    if t["type"] == "client.action" and t["name"] == "set_mode":
        requested = (t.get("args") or {}).get("mode")
        if requested not in ("planning", "executing"):
            return {"state": state}
        entering = requested != state["mode"]
        state["mode"] = requested
        if entering and requested == "executing":
            seed = {"role": "user", "content": render_plan(state["plan"])}
            return {"messages": [seed], "actions": [prompt([seed], "executing")], "state": state}
        return {"state": state}

    # The client sent the conversation → record it, prompt the model in this mode.
    if t["type"] == "client.messages":
        return {"messages": t["messages"], "actions": [prompt(t["messages"], state["mode"])], "state": state}

    # A declared tool call with valid arguments → mutate the plan, answer.
    if t["type"] == "tool.execute":
        tool = ALL_TOOLS[t["name"]]
        return {"actions": [{"type": "tool.result", "result": tool["exec"](state["plan"], t["input"]["value"])}], "state": state}

    return {"state": state}
