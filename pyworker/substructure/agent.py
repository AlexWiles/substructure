"""Agent handle with decorator-based registration for tools and event handlers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Union, get_type_hints, get_origin, get_args

_PYTHON_TYPE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# Sentinel for NoneType
_NoneType = type(None)


def _type_to_json_schema(tp: Any) -> tuple[dict, bool]:
    """Convert a Python type annotation to a JSON schema dict.

    Returns (schema_dict, is_optional).
    """
    origin = get_origin(tp)
    args = get_args(tp)

    # Handle Optional[X] / X | None  (Union[X, None])
    if origin is Union:
        non_none = [a for a in args if a is not _NoneType]
        if len(non_none) == 1:
            schema, _ = _type_to_json_schema(non_none[0])
            return schema, True
        # Union of multiple non-None types — not representable cleanly
        return {"type": "string"}, _NoneType in args

    # Handle list[X]
    if origin is list:
        if args:
            items, _ = _type_to_json_schema(args[0])
            return {"type": "array", "items": items}, False
        return {"type": "array"}, False

    # Handle dict[K, V]
    if origin is dict:
        return {"type": "object"}, False

    # Plain types
    json_type = _PYTHON_TYPE_TO_JSON.get(tp, "string")
    return {"type": json_type}, False


def _function_to_tool_schema(fn: Callable) -> dict:
    """Generate a JSON-schema-style tool definition from a function signature."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    doc = inspect.getdoc(fn) or ""

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        tp = hints.get(name, str)
        schema, is_optional = _type_to_json_schema(tp)
        properties[name] = schema
        if param.default is inspect.Parameter.empty and not is_optional:
            required.append(name)

    return {
        "name": fn.__name__,
        "description": doc,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


@dataclass
class ToolDef:
    fn: Callable
    schema: dict


@dataclass
class SubAgentDef:
    agent_name: str
    schema: dict


class AgentHandle:
    """Returned by ``Worker.agent()``. Use decorators to register handlers and tools."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, ToolDef] = {}
        self.subagents: dict[str, SubAgentDef] = {}
        self.handlers: dict[str, Callable] = {}

    # -- tool / subagent decorators ------------------------------------------

    def tool(self, fn: Callable) -> Callable:
        """Register an async function as a tool the agent can invoke."""
        self.tools[fn.__name__] = ToolDef(fn=fn, schema=_function_to_tool_schema(fn))
        return fn

    def subagent(self, agent_name: str) -> Callable:
        """Declare a sub-agent delegation tool.  The function body is unused."""
        def decorator(fn: Callable) -> Callable:
            self.subagents[fn.__name__] = SubAgentDef(
                agent_name=agent_name,
                schema=_function_to_tool_schema(fn),
            )
            return fn
        return decorator

    # -- event handler decorators --------------------------------------------

    def on_user_message(self, fn: Callable) -> Callable:
        self.handlers["user_message"] = fn
        return fn

    def on_llm_completed(self, fn: Callable) -> Callable:
        self.handlers["llm_completed"] = fn
        return fn

    def on_llm_failed(self, fn: Callable) -> Callable:
        self.handlers["llm_failed"] = fn
        return fn

    def on_tool_resolved(self, fn: Callable) -> Callable:
        self.handlers["tool_resolved"] = fn
        return fn

    def on_interrupt_resumed(self, fn: Callable) -> Callable:
        self.handlers["interrupt_resumed"] = fn
        return fn

    def on_stall(self, fn: Callable) -> Callable:
        self.handlers["stall"] = fn
        return fn

    # -- helpers -------------------------------------------------------------

    @property
    def all_tools(self) -> list[dict]:
        """Combined tool schemas for tools + sub-agent declarations."""
        return [t.schema for t in self.tools.values()] + [
            s.schema for s in self.subagents.values()
        ]
