"""Protocol types matching the substructure2 Rust runtime JSON wire format."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# --- Messages ---


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class Message(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: Role
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


# --- LLM request ---


class LlmToolFunction(BaseModel):
    name: str
    description: str
    parameters: Any


class LlmTool(BaseModel):
    function: LlmToolFunction


class LlmRequest(BaseModel):
    model: str
    messages: list[Message]
    tools: Optional[list[LlmTool]] = None
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None


# --- Tool results ---


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


# --- Artifacts ---


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str


class DataPart(BaseModel):
    kind: Literal["data"] = "data"
    data: Any


Part = Annotated[Union[TextPart, DataPart], Field(discriminator="kind")]


class Artifact(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parts: list[Part] = []


# --- Retry policy ---


class RetryPolicy(BaseModel):
    timeout_secs: Optional[int] = None
    max_retries: int = 0
    backoff_base_secs: int = 0
    backoff_max_secs: int = 0

    @classmethod
    def no_retry(cls) -> RetryPolicy:
        return cls()

    @classmethod
    def default(cls) -> RetryPolicy:
        return cls(timeout_secs=30, max_retries=3, backoff_base_secs=2, backoff_max_secs=60)


# --- Tool handler ---


class ToolHandler(str, Enum):
    worker = "worker"
    client = "client"
    sub_agent = "sub_agent"


# --- Decision trigger (tagged union, discriminator = "type") ---


class UserMessageTrigger(BaseModel):
    type: Literal["user_message"] = "user_message"
    stream: bool
    message: Message


class LlmCompletedTrigger(BaseModel):
    type: Literal["llm_completed"] = "llm_completed"
    call_id: str
    message: Message
    truncated: bool = False


class LlmFailedTrigger(BaseModel):
    type: Literal["llm_failed"] = "llm_failed"
    call_id: str
    error: str


class ToolResolvedTrigger(BaseModel):
    type: Literal["tool_resolved"] = "tool_resolved"
    result: ToolResult


class InterruptResumedTrigger(BaseModel):
    type: Literal["interrupt_resumed"] = "interrupt_resumed"
    interrupt_id: str


class StallTrigger(BaseModel):
    type: Literal["stall"] = "stall"


DecisionTrigger = Annotated[
    Union[
        UserMessageTrigger,
        LlmCompletedTrigger,
        LlmFailedTrigger,
        ToolResolvedTrigger,
        InterruptResumedTrigger,
        StallTrigger,
    ],
    Field(discriminator="type"),
]


# --- Worker actions (tagged union, discriminator = "type") ---


class RequestLlmAction(BaseModel):
    type: Literal["request_llm"] = "request_llm"
    request: LlmRequest
    stream: bool = False
    llm_client: str = ""
    retry: RetryPolicy = Field(default_factory=RetryPolicy.default)


class RequestToolCallAction(BaseModel):
    type: Literal["request_tool_call"] = "request_tool_call"
    tool_call_id: str
    name: str
    arguments: str
    handler: ToolHandler = ToolHandler.worker
    retry: RetryPolicy = Field(default_factory=RetryPolicy.no_retry)


class RequestSubAgentAction(BaseModel):
    type: Literal["request_sub_agent"] = "request_sub_agent"
    agent_id: str
    message: str
    retry: RetryPolicy = Field(default_factory=RetryPolicy.default)


class DoneAction(BaseModel):
    type: Literal["done"] = "done"
    artifacts: list[Artifact] = []


WorkerAction = Annotated[
    Union[RequestLlmAction, RequestToolCallAction, RequestSubAgentAction, DoneAction],
    Field(discriminator="type"),
]


# --- Span context ---


class SpanContext(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 1
    trace_state: Optional[str] = None
    name: Optional[str] = None


# --- HTTP wire types ---


class PollRequest(BaseModel):
    tenant_id: str
    agent_ids: list[str]
    timeout_ms: int = 30_000


class PollResponse(BaseModel):
    session_id: UUID
    tenant_id: str
    decision_id: str
    agent_id: str
    trigger: DecisionTrigger
    worker_state: str  # base64-encoded bytes
    span: SpanContext
    attempts: int = 0
    deadline: Optional[datetime] = None


class SubmitRequest(BaseModel):
    session_id: UUID
    tenant_id: str
    decision_id: str
    actions: list[WorkerAction]
    state: str  # base64-encoded bytes
    span: Optional[SpanContext] = None


class SubmitResponse(BaseModel):
    ok: bool
    error: Optional[str] = None


# --- Client API types ---


class SendMessageRequest(BaseModel):
    agent_id: str
    message: str
    tenant_id: Optional[str] = None
    session_id: Optional[UUID] = None


class SendMessageResponse(BaseModel):
    session_id: UUID
    ok: bool
    error: Optional[str] = None
