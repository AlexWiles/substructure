"""Session and AgentContext — passed into every handler."""

from __future__ import annotations

import json
from typing import Any

from google.protobuf import json_format, struct_pb2

from substructure._proto import worker_pb2

# ---------------------------------------------------------------------------
# Proto ↔ dict helpers
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "system": worker_pb2.ROLE_SYSTEM,
    "user": worker_pb2.ROLE_USER,
    "assistant": worker_pb2.ROLE_ASSISTANT,
    "tool": worker_pb2.ROLE_TOOL,
}
_ROLE_NAMES = {v: k for k, v in _ROLE_MAP.items()}

_CALL_STATUS_NAMES = {
    0: "UNSPECIFIED",
    1: "PENDING",
    2: "COMPLETED",
    3: "FAILED",
    4: "RETRY_SCHEDULED",
}


def _dict_to_message(d: dict) -> worker_pb2.Message:
    kwargs: dict[str, Any] = {
        "role": _ROLE_MAP.get(d.get("role", ""), worker_pb2.ROLE_UNSPECIFIED),
    }
    if d.get("content") is not None:
        kwargs["content"] = d["content"]
    if d.get("tool_call_id") is not None:
        kwargs["tool_call_id"] = d["tool_call_id"]
    if d.get("call_id") is not None:
        kwargs["call_id"] = d["call_id"]
    tcs = d.get("tool_calls")
    if tcs:
        kwargs["tool_calls"] = [
            worker_pb2.ToolCall(
                id=tc["id"], name=tc["name"], arguments=tc.get("arguments", "{}"),
            )
            for tc in tcs
        ]
    return worker_pb2.Message(**kwargs)


def _message_to_dict(msg: worker_pb2.Message) -> dict:
    d: dict[str, Any] = {"role": _ROLE_NAMES.get(msg.role, "unknown")}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    if msg.call_id:
        d["call_id"] = msg.call_id
    return d


def _dict_to_llm_tool(schema: dict) -> worker_pb2.LlmTool:
    params = schema.get("parameters")
    params_value = None
    if params:
        params_value = json_format.ParseDict(params, struct_pb2.Value())
    return worker_pb2.LlmTool(
        tool_type="function",
        function=worker_pb2.LlmToolFunction(
            name=schema["name"],
            description=schema.get("description", ""),
            parameters=params_value,
        ),
    )


def _tool_result_to_dict(r: worker_pb2.ToolResult) -> dict:
    return {
        "tool_call_id": r.tool_call_id,
        "name": r.name,
        "content": r.content,
        "is_error": r.is_error,
    }


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


class AgentContext:
    """Read-only metadata about the current dispatch, passed to every handler."""

    __slots__ = (
        "session_id",
        "agent_name",
        "stream",
        "token_usage",
        "tool_call_statuses",
        "llm_call_statuses",
        "all_tools",
    )

    def __init__(
        self,
        dispatch: worker_pb2.WorkerDispatch,
        all_tools: list[dict],
    ) -> None:
        self.session_id: str = dispatch.session_id
        self.agent_name: str = dispatch.agent_name
        self.stream: bool = dispatch.stream
        self.token_usage: dict[str, int] = dict(dispatch.token_usage)
        self.tool_call_statuses: dict[str, str] = {
            k: _CALL_STATUS_NAMES.get(v, "UNSPECIFIED")
            for k, v in dispatch.tool_call_statuses.items()
        }
        self.llm_call_statuses: dict[str, str] = {
            k: _CALL_STATUS_NAMES.get(v, "UNSPECIFIED")
            for k, v in dispatch.llm_call_statuses.items()
        }
        self.all_tools = all_tools


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """Wraps opaque worker state and provides action builder methods."""

    def __init__(self, state_bytes: bytes) -> None:
        self._state: dict | None = (
            json.loads(state_bytes) if state_bytes else None
        )

    @property
    def state(self) -> dict | None:
        return self._state

    # -- action builders -----------------------------------------------------

    def decide(
        self,
        actions: list[worker_pb2.WorkerAction],
        state: dict | None = None,
    ) -> worker_pb2.WorkerDecision:
        raw = json.dumps(state).encode() if state is not None else b""
        return worker_pb2.WorkerDecision(actions=actions, state=raw)

    def request_llm(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
        client: str = "",
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        timeout_secs: int | None = None,
        max_retries: int | None = None,
    ) -> worker_pb2.WorkerAction:
        llm_req = worker_pb2.LlmRequest(
            model=model,
            messages=[_dict_to_message(m) for m in messages],
            tools=[_dict_to_llm_tool(t) for t in (tools or [])],
        )
        if temperature is not None:
            llm_req.temperature = temperature
        if max_completion_tokens is not None:
            llm_req.max_completion_tokens = max_completion_tokens
        return worker_pb2.WorkerAction(
            request_llm=worker_pb2.RequestLlm(
                request=llm_req,
                stream=stream,
                llm_client=client,
                timeout_secs=timeout_secs,
                max_retries=max_retries,
            ),
        )

    def request_tool_calls(
        self, tool_calls: list[dict],
    ) -> worker_pb2.WorkerAction:
        return worker_pb2.WorkerAction(
            request_tool_calls=worker_pb2.RequestToolCalls(
                tool_calls=[
                    worker_pb2.ToolCallAction(
                        tool_call=worker_pb2.ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=tc.get("arguments", "{}"),
                        ),
                    )
                    for tc in tool_calls
                ],
            ),
        )

    def request_sub_agent(
        self, agent_name: str, message: str,
    ) -> worker_pb2.WorkerAction:
        return worker_pb2.WorkerAction(
            request_sub_agent=worker_pb2.RequestSubAgent(
                agent_name=agent_name,
                message=message,
            ),
        )

    def done(
        self,
        text: str | None = None,
        error: str | None = None,
    ) -> worker_pb2.WorkerAction:
        content = error or text
        artifacts = []
        if content:
            artifacts.append(
                worker_pb2.Artifact(parts=[worker_pb2.Part(text=content)])
            )
        return worker_pb2.WorkerAction(
            done=worker_pb2.Done(artifacts=artifacts),
        )
