"""Substructure client — single async gRPC connection for both worker and session RPCs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator

import grpc.aio

from substructure._proto import worker_pb2, worker_pb2_grpc
from substructure.worker import Worker

logger = logging.getLogger(__name__)


@dataclass
class SessionEvent:
    """A single event from a running session."""

    session_id: str
    event_type: str
    sequence: int
    payload: dict

    @property
    def is_done(self) -> bool:
        return self.event_type == "session.done"


class Substructure:
    """Top-level entry point. Wraps a single async gRPC channel to the gateway.

    Usage::

        client = Substructure("localhost:50051")

        # Create a worker
        worker = client.worker("my-worker")
        echo = worker.agent("echo")
        # ... register handlers ...
        # asyncio.create_task(worker.start())

        # Run a session
        async for event in client.run_session(agent="echo", message="hello"):
            print(event.event_type, event.payload)
            if event.is_done:
                break
    """

    def __init__(self, endpoint: str = "localhost:50051", token: str | None = None) -> None:
        self._endpoint = endpoint
        self._channel = grpc.aio.insecure_channel(endpoint)
        self._stub = worker_pb2_grpc.WorkerGatewayStub(self._channel)
        self._metadata: list[tuple[str, str]] = []
        if token:
            self._metadata.append(("authorization", f"Bearer {token}"))

    def worker(self, worker_id: str = "pyworker-0") -> Worker:
        """Create a Worker that shares this connection."""
        return Worker(
            stub=self._stub,
            metadata=self._metadata,
            worker_id=worker_id,
        )

    async def run_session(self, agent: str, message: str) -> AsyncIterator[SessionEvent]:
        """Create a session, send a message, and yield events until completion."""
        request = worker_pb2.RunSessionRequest(agent=agent, message=message)
        stream = self._stub.RunSession(request, metadata=self._metadata)

        async for proto_event in stream:
            payload = {}
            if proto_event.payload_json:
                try:
                    payload = json.loads(proto_event.payload_json)
                except json.JSONDecodeError:
                    payload = {"raw": proto_event.payload_json}

            yield SessionEvent(
                session_id=proto_event.session_id,
                event_type=proto_event.event_type,
                sequence=proto_event.sequence,
                payload=payload,
            )

    async def close(self) -> None:
        await self._channel.close()

    async def __aenter__(self) -> Substructure:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
