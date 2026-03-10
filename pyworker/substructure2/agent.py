"""Agent definition with PydanticAI-style decorator API."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import UUID

from ._schema import function_to_tool
from .types import (
    Artifact,
    DecisionTrigger,
    DoneAction,
    LlmCompletedTrigger,
    LlmFailedTrigger,
    LlmRequest,
    LlmTool,
    Message,
    RequestLlmAction,
    RetryPolicy,
    Role,
    SpanContext,
    StallTrigger,
    TextPart,
    ToolCall,
    ToolCallFunction,
    ToolResolvedTrigger,
    ToolResult,
    UserMessageTrigger,
    WorkerAction,
)

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    """Context passed to tool functions and trigger handlers."""

    session_id: UUID
    tenant_id: str
    agent_id: str
    decision_id: str
    attempts: int = 0
    span: Optional[SpanContext] = None


@dataclass
class _ToolDef:
    """Internal tool registration."""

    fn: Callable[..., Any]
    tool_spec: LlmTool
    needs_context: bool


@dataclass
class _AgentState:
    """Serializable conversation state carried in worker_state bytes."""

    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_bytes(self) -> bytes:
        return json.dumps({"messages": self.messages}).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> _AgentState:
        if not data:
            return cls()
        parsed = json.loads(data)
        return cls(messages=parsed.get("messages", []))

    def append(self, msg: Message) -> None:
        self.messages.append(msg.model_dump(exclude_none=True))

    def to_messages(self) -> list[Message]:
        return [Message.model_validate(m) for m in self.messages]


# Type alias for trigger handler return
TriggerHandlerResult = tuple[list[WorkerAction], bytes]

# Custom handler signature: (ctx, trigger) -> (actions, state_bytes)
TriggerHandler = Callable[..., Any]


class Agent:
    """Defines an agent with tools and optional custom trigger handlers.

    Usage::

        agent = Agent("my_agent", model="gpt-4o", instructions="Be helpful.")

        @agent.tool
        async def search(ctx: RunContext, query: str) -> str:
            \"\"\"Search the web.\"\"\"
            return f"Results for {query}"

        @agent.tool_plain
        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b
    """

    def __init__(
        self,
        name: str,
        *,
        model: str = "gpt-4o",
        instructions: str = "",
        llm_client: str = "",
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        retry: Optional[RetryPolicy] = None,
    ) -> None:
        self.name = name
        self.model = model
        self.instructions = instructions
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.retry = retry or RetryPolicy.default()
        self._tools: dict[str, _ToolDef] = {}
        self._handlers: dict[str, TriggerHandler] = {}

    # --- Decorators ---

    def tool(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a tool function. First parameter must be RunContext."""
        spec = function_to_tool(fn, skip_first=True)
        self._tools[fn.__name__] = _ToolDef(fn=fn, tool_spec=spec, needs_context=True)
        return fn

    def tool_plain(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a tool function without RunContext."""
        spec = function_to_tool(fn, skip_first=False)
        self._tools[fn.__name__] = _ToolDef(fn=fn, tool_spec=spec, needs_context=False)
        return fn

    def on(self, trigger_type: str) -> Callable[[TriggerHandler], TriggerHandler]:
        """Register a custom handler for a trigger type.

        Usage::

            @agent.on("user_message")
            async def handle(ctx: RunContext, trigger: UserMessageTrigger) -> list[WorkerAction]:
                ...
        """

        def decorator(fn: TriggerHandler) -> TriggerHandler:
            self._handlers[trigger_type] = fn
            return fn

        return decorator

    # --- Tool specs ---

    def _llm_tools(self) -> list[LlmTool] | None:
        if not self._tools:
            return None
        return [td.tool_spec for td in self._tools.values()]

    # --- Decision entry point ---

    async def decide(
        self,
        trigger: DecisionTrigger,
        worker_state: bytes,
        ctx: RunContext,
    ) -> tuple[list[WorkerAction], bytes]:
        """Process a decision trigger and return actions + updated state.

        If a custom handler is registered for the trigger type, it is called.
        Otherwise the default agentic loop handles it.
        """
        trigger_type = trigger.type  # type: ignore[union-attr]

        # Custom handler?
        handler = self._handlers.get(trigger_type)
        if handler is not None:
            result = handler(ctx, trigger)
            if inspect.isawaitable(result):
                result = await result
            # Handler returns (actions, state_bytes) or just actions
            if isinstance(result, tuple):
                return result
            return result, worker_state

        # Default loop
        return await self._default_decide(trigger, worker_state, ctx)

    async def _default_decide(
        self,
        trigger: DecisionTrigger,
        worker_state: bytes,
        ctx: RunContext,
    ) -> tuple[list[WorkerAction], bytes]:
        state = _AgentState.from_bytes(worker_state)

        match trigger:
            case UserMessageTrigger():
                logger.info("User: %s", (trigger.message.content or "")[:120])
                state.append(trigger.message)
                return self._request_llm(state), state.to_bytes()

            case LlmCompletedTrigger():
                state.append(trigger.message)

                # If tool calls, execute locally and loop
                if trigger.message.tool_calls:
                    names = [tc.function.name for tc in trigger.message.tool_calls]
                    logger.info("LLM requested tools: %s", names)
                    await self._execute_tools(trigger.message.tool_calls, state, ctx)
                    return self._request_llm(state), state.to_bytes()

                # No tool calls — done
                text = trigger.message.content or ""
                logger.info("LLM response: %s", text[:200])
                return self._done(text), state.to_bytes()

            case LlmFailedTrigger():
                logger.error("LLM failed: %s", trigger.error)
                return self._done(f"LLM error: {trigger.error}"), state.to_bytes()

            case ToolResolvedTrigger():
                # Tool was resolved by the runtime (e.g. handler=client)
                state.append(Message(
                    role=Role.tool,
                    content=trigger.result.content,
                    tool_call_id=trigger.result.tool_call_id,
                ))
                return self._request_llm(state), state.to_bytes()

            case StallTrigger():
                return self._request_llm(state), state.to_bytes()

            case _:
                logger.warning("Unhandled trigger type: %s", type(trigger).__name__)
                return self._request_llm(state), state.to_bytes()

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        state: _AgentState,
        ctx: RunContext,
    ) -> None:
        """Execute tool calls locally and append results to state."""
        tasks = []
        for tc in tool_calls:
            tasks.append(self._execute_one_tool(tc, ctx))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tc, result in zip(tool_calls, results):
            if isinstance(result, BaseException):
                content = f"Error: {result}"
                logger.error("Tool %s failed: %s", tc.function.name, result)
            else:
                content = str(result)
                logger.info("Tool %s -> %s", tc.function.name, content[:120])

            state.append(Message(
                role=Role.tool,
                content=content,
                tool_call_id=tc.id,
            ))

    async def _execute_one_tool(self, tc: ToolCall, ctx: RunContext) -> Any:
        tool_def = self._tools.get(tc.function.name)
        if tool_def is None:
            raise ValueError(f"Unknown tool: {tc.function.name}")

        kwargs = json.loads(tc.function.arguments) if tc.function.arguments else {}

        if tool_def.needs_context:
            result = tool_def.fn(ctx, **kwargs)
        else:
            result = tool_def.fn(**kwargs)

        if inspect.isawaitable(result):
            result = await result

        return result

    def _request_llm(self, state: _AgentState) -> list[WorkerAction]:
        messages = state.to_messages()
        if self.instructions:
            messages = [Message(role=Role.system, content=self.instructions)] + messages

        request = LlmRequest(
            model=self.model,
            messages=messages,
            tools=self._llm_tools(),
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
        )
        return [
            RequestLlmAction(
                request=request,
                llm_client=self.llm_client,
                retry=self.retry,
            )
        ]

    def _done(self, text: str) -> list[WorkerAction]:
        artifacts = []
        if text:
            artifacts.append(Artifact(parts=[TextPart(text=text)]))
        return [DoneAction(artifacts=artifacts)]
