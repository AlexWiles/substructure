"""Generate JSON schemas from Python function signatures for LLM tool specs."""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Optional, Union, get_args, get_origin, get_type_hints

from .types import LlmTool, LlmToolFunction

# Maps Python types to JSON Schema types.
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _type_to_schema(tp: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    origin = get_origin(tp)

    # Optional[X] → nullable X
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            schema = _type_to_schema(args[0])
            return schema
        return {}

    # list[X]
    if origin is list:
        args = get_args(tp)
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _type_to_schema(args[0])
        return schema

    # dict[K, V]
    if origin is dict:
        args = get_args(tp)
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = _type_to_schema(args[1])
        return schema

    # Pydantic models
    if isinstance(tp, type) and hasattr(tp, "model_json_schema"):
        return tp.model_json_schema()

    # Primitive types
    if tp in _TYPE_MAP:
        return {"type": _TYPE_MAP[tp]}

    # Enum
    if isinstance(tp, type) and issubclass(tp, __import__("enum").Enum):
        return {"type": "string", "enum": [e.value for e in tp]}

    return {}


def _parse_docstring_params(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from Google-style docstrings.

    Supports:
        Args:
            param: Description text.
            param: Description text
                that spans multiple lines.
    """
    if not docstring:
        return {}

    params: dict[str, str] = {}
    in_args = False
    current_param: str | None = None
    current_desc: list[str] = []

    for line in docstring.split("\n"):
        stripped = line.strip()

        if stripped.lower().startswith("args:"):
            in_args = True
            continue

        if in_args:
            # New section header ends Args block
            if stripped and not stripped.startswith("-") and ":" in stripped:
                match = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
                if match:
                    # Save previous param
                    if current_param:
                        params[current_param] = " ".join(current_desc).strip()
                    current_param = match.group(1)
                    current_desc = [match.group(2)] if match.group(2) else []
                    continue

            # Section break (e.g. "Returns:", "Raises:", blank line after content)
            if stripped and re.match(r"^[A-Z]\w*:", stripped) and current_param:
                params[current_param] = " ".join(current_desc).strip()
                in_args = False
                current_param = None
                continue

            # Continuation line
            if current_param and stripped:
                current_desc.append(stripped)
                continue

            # Blank line after params
            if not stripped and current_param:
                params[current_param] = " ".join(current_desc).strip()
                current_param = None

    # Flush last param
    if current_param:
        params[current_param] = " ".join(current_desc).strip()

    return params


def _get_description(fn: Callable[..., Any]) -> str:
    """Extract the summary line from a function's docstring."""
    doc = inspect.getdoc(fn)
    if not doc:
        return fn.__name__
    # First non-empty line
    for line in doc.split("\n"):
        line = line.strip()
        if line:
            return line
    return fn.__name__


def function_schema(
    fn: Callable[..., Any],
    *,
    skip_first: bool = False,
) -> LlmToolFunction:
    """Build an LlmToolFunction from a Python function.

    Args:
        fn: The function to inspect.
        skip_first: If True, skip the first parameter (e.g. RunContext).
    """
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn)
    param_docs = _parse_docstring_params(doc)

    properties: dict[str, Any] = {}
    required: list[str] = []

    params = list(sig.parameters.values())
    if skip_first and params:
        params = params[1:]

    for param in params:
        if param.name in ("self", "cls"):
            continue

        tp = hints.get(param.name, str)
        schema = _type_to_schema(tp)

        if param.name in param_docs:
            schema["description"] = param_docs[param.name]

        properties[param.name] = schema

        # Required if no default value and not Optional
        is_optional = (
            get_origin(tp) is Union
            and type(None) in get_args(tp)
        )
        if param.default is inspect.Parameter.empty and not is_optional:
            required.append(param.name)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return LlmToolFunction(
        name=fn.__name__,
        description=_get_description(fn),
        parameters=parameters,
    )


def function_to_tool(
    fn: Callable[..., Any],
    *,
    skip_first: bool = False,
) -> LlmTool:
    """Build a complete LlmTool from a Python function."""
    return LlmTool(function=function_schema(fn, skip_first=skip_first))
