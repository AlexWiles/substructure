"""Worker — pulls work from the gateway and dispatches to registered agent handlers."""

from __future__ import annotations

import json
import logging
from typing import Any

from substructure._proto import worker_pb2, worker_pb2_grpc
from substructure.agent import AgentHandle
from substructure.session import AgentContext, Session, _message_to_dict, _tool_result_to_dict

logger = logging.getLogger(__name__)


class Worker:
    """Created via ``Substructure.worker()``. Register agents, then call ``await start()``."""

    def __init__(
        self,
        stub: worker_pb2_grpc.WorkerGatewayStub,
        metadata: list[tuple[str, str]],
        worker_id: str = "pyworker-0",
    ) -> None:
        self._stub = stub
        self._metadata = metadata
        self._worker_id = worker_id
        self._agents: dict[str, AgentHandle] = {}

    def agent(self, name: str) -> AgentHandle:
        """Register and return a handle for the named agent."""
        handle = AgentHandle(name)
        self._agents[name] = handle
        return handle

    # -- public entry --------------------------------------------------------

    async def start(self) -> None:
        """Pull work from the gateway in a loop and dispatch it. Runs until cancelled."""
        agent_names = list(self._agents.keys())

        logger.info(
            "worker %s starting for agents %s",
            self._worker_id,
            agent_names,
        )

        pending_result: worker_pb2.WorkResult | None = None

        while True:
            request = worker_pb2.GetWorkRequest(
                worker_id=self._worker_id,
                agent_names=agent_names,
            )
            if pending_result is not None:
                request.result.CopyFrom(pending_result)

            item = await self._stub.GetWork(request, metadata=self._metadata)
            pending_result = await self._execute(item)

    # -- dispatch ------------------------------------------------------------

    async def _execute(self, item: worker_pb2.WorkItem) -> worker_pb2.WorkResult | None:
        which = item.WhichOneof("item")

        if which == "decision":
            return await self._handle_decision(item.decision)
        elif which == "tool_call":
            return await self._handle_tool_call(item.tool_call)
        else:
            logger.warning("empty work item, skipping")
            return None

    async def _handle_decision(
        self, dispatch: worker_pb2.WorkerDispatch,
    ) -> worker_pb2.WorkResult:
        agent = self._agents.get(dispatch.agent_name)
        session = Session(dispatch.worker_state)

        if agent is None:
            decision = session.decide(
                actions=[session.done(error=f"unknown agent: {dispatch.agent_name}")],
                state=session.state,
            )
        else:
            ctx = AgentContext(dispatch, all_tools=agent.all_tools)
            decision = await self._dispatch_trigger(agent, session, ctx, dispatch)

        return worker_pb2.WorkResult(
            decision=worker_pb2.DecisionResult(
                session_id=dispatch.session_id,
                decision_id=dispatch.decision_id,
                decision=decision,
                span=dispatch.span,
            ),
        )

    async def _dispatch_trigger(
        self,
        agent: AgentHandle,
        session: Session,
        ctx: AgentContext,
        dispatch: worker_pb2.WorkerDispatch,
    ) -> worker_pb2.WorkerDecision:
        trigger = dispatch.trigger
        which = trigger.WhichOneof("trigger")

        handler = agent.handlers.get({
            "user_message": "user_message",
            "llm_completed": "llm_completed",
            "llm_failed": "llm_failed",
            "tool_resolved": "tool_resolved",
            "interrupt_resumed": "interrupt_resumed",
            "stall": "stall",
        }.get(which, ""))

        if handler is None:
            return session.decide(actions=[session.done()], state=session.state)

        if which == "user_message":
            um = trigger.user_message
            msg_content = um.message.content if um.message else ""
            return await handler(session, ctx, msg_content, um.stream)

        elif which == "llm_completed":
            lc = trigger.llm_completed
            msg_dict = _message_to_dict(lc.message) if lc.message else {}
            return await handler(session, ctx, lc.call_id, msg_dict, lc.truncated)

        elif which == "llm_failed":
            lf = trigger.llm_failed
            return await handler(session, ctx, lf.call_id, lf.error)

        elif which == "tool_resolved":
            tr = trigger.tool_resolved
            result_dict = _tool_result_to_dict(tr.result) if tr.result else {}
            return await handler(session, ctx, result_dict)

        elif which == "interrupt_resumed":
            ir = trigger.interrupt_resumed
            return await handler(session, ctx, ir.interrupt_id)

        elif which == "stall":
            return await handler(session, ctx)

        return session.decide(actions=[session.done()], state=session.state)

    async def _handle_tool_call(
        self, dispatch: worker_pb2.ToolCallDispatch,
    ) -> worker_pb2.WorkResult:
        agent = self._agents.get(dispatch.agent_name)
        tool_def = agent.tools.get(dispatch.name) if agent else None

        if tool_def is None:
            result = worker_pb2.ToolCallResult(
                tool_call_id=dispatch.tool_call_id,
                name=dispatch.name,
                error=f"unknown tool: {dispatch.name}",
            )
        else:
            try:
                args: dict[str, Any] = json.loads(dispatch.arguments) if dispatch.arguments else {}
                value = await tool_def.fn(**args)
                result = worker_pb2.ToolCallResult(
                    tool_call_id=dispatch.tool_call_id,
                    name=dispatch.name,
                    result=str(value),
                )
            except Exception as exc:
                result = worker_pb2.ToolCallResult(
                    tool_call_id=dispatch.tool_call_id,
                    name=dispatch.name,
                    error=str(exc),
                )

        return worker_pb2.WorkResult(
            tool_call=worker_pb2.ToolCallSubmission(
                session_id=dispatch.session_id,
                result=result,
                span=dispatch.span,
            ),
        )
