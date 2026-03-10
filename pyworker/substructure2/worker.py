"""Worker: HTTP poll/submit loop against a substructure2 runtime."""

from __future__ import annotations

import asyncio
import logging
import signal
from base64 import b64decode, b64encode
from typing import Optional
from uuid import UUID

import httpx

from .agent import Agent, RunContext
from .client import Client
from .types import (
    PollRequest,
    PollResponse,
    SendMessageResponse,
    SpanContext,
    SubmitRequest,
    SubmitResponse,
    WorkerAction,
)

logger = logging.getLogger(__name__)


class Worker:
    """Connects to a substructure2 runtime and runs agents via HTTP polling.

    Usage::

        worker = Worker(
            url="http://localhost:3000",
            tenant_id="acme",
            agents=[agent],
        )
        asyncio.run(worker.run())
    """

    def __init__(
        self,
        client: Client,
        *,
        agents: list[Agent],
        poll_timeout_ms: int = 30_000,
    ) -> None:
        self.client = client
        self.url = client.url
        self.tenant_id = client.tenant_id
        self.poll_timeout_ms = poll_timeout_ms
        self._agents: dict[str, Agent] = {a.name: a for a in agents}
        self._running = True

    async def send(
        self,
        agent_id: str,
        message: str,
        *,
        session_id: Optional[UUID] = None,
    ) -> SendMessageResponse:
        """Send a message to an agent, creating a session if needed."""
        return await self.client.send(agent_id, message, session_id=session_id)

    async def run(self) -> None:
        """Main loop: poll for work, dispatch to agents, submit results."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown)

        agent_ids = list(self._agents.keys())
        logger.info(
            "Worker starting: url=%s tenant=%s agents=%s",
            self.url,
            self.tenant_id,
            agent_ids,
        )

        async with httpx.AsyncClient(base_url=self.url, timeout=60.0) as client:
            while self._running:
                try:
                    await self._poll_cycle(client, agent_ids)
                except httpx.HTTPError as e:
                    logger.error("HTTP error during poll: %s", e)
                    await asyncio.sleep(1.0)
                except Exception:
                    logger.exception("Unexpected error in poll loop")
                    await asyncio.sleep(1.0)

        logger.info("Worker stopped.")

    async def _poll_cycle(
        self,
        client: httpx.AsyncClient,
        agent_ids: list[str],
    ) -> None:
        poll_req = PollRequest(
            tenant_id=self.tenant_id,
            agent_ids=agent_ids,
            timeout_ms=self.poll_timeout_ms,
        )

        resp = await client.post(
            "/workers/poll",
            json=poll_req.model_dump(exclude_none=True, mode="json"),
        )

        if resp.status_code == 204:
            logger.debug("Poll: no work available")
            return

        resp.raise_for_status()
        poll = PollResponse.model_validate(resp.json())
        trigger_type = poll.trigger.type  # type: ignore[union-attr]
        logger.info(
            "Received %s for agent=%s session=%s decision=%s",
            trigger_type,
            poll.agent_id,
            poll.session_id,
            poll.decision_id,
        )

        agent = self._agents.get(poll.agent_id)
        if agent is None:
            logger.error("No agent registered for %s", poll.agent_id)
            return

        ctx = RunContext(
            session_id=poll.session_id,
            tenant_id=poll.tenant_id,
            agent_id=poll.agent_id,
            decision_id=poll.decision_id,
            attempts=poll.attempts,
            span=poll.span,
        )

        worker_state = b64decode(poll.worker_state) if poll.worker_state else b""

        try:
            actions, new_state = await agent.decide(poll.trigger, worker_state, ctx)
        except Exception:
            logger.exception(
                "Agent %s failed on decision %s",
                poll.agent_id,
                poll.decision_id,
            )
            return

        action_types = [a.type for a in actions]  # type: ignore[union-attr]
        logger.info(
            "Submitting actions=%s for decision=%s",
            action_types,
            poll.decision_id,
        )

        submit_req = SubmitRequest(
            session_id=poll.session_id,
            tenant_id=poll.tenant_id,
            decision_id=poll.decision_id,
            actions=actions,
            state=b64encode(new_state).decode(),
            span=poll.span,
        )

        submit_resp = await client.post(
            "/workers/submit",
            json=submit_req.model_dump(exclude_none=True, mode="json"),
        )
        submit_resp.raise_for_status()

        result = SubmitResponse.model_validate(submit_resp.json())
        if not result.ok:
            logger.error(
                "Submit rejected for decision %s: %s",
                poll.decision_id,
                result.error,
            )

    def _shutdown(self) -> None:
        logger.info("Shutdown signal received.")
        self._running = False
