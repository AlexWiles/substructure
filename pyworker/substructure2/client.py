"""Client for sending messages to a substructure2 runtime."""

from __future__ import annotations

import asyncio
import json
from queue import Queue
from threading import Thread
from typing import AsyncIterator, Iterator, Optional
from uuid import UUID

import httpx

from .types import SendMessageRequest


class Client:
    """Send messages to agents on a substructure2 runtime.

    Usage::

        client = Client("http://localhost:8080")
        for event in client.send("weather", "What's the weather in NYC?"):
            print(event)
    """

    def __init__(self, url: str, *, tenant_id: str = "default") -> None:
        self.url = url.rstrip("/")
        self.tenant_id = tenant_id

    def send(
        self,
        agent_id: str,
        message: str,
        *,
        session_id: Optional[UUID] = None,
    ) -> Iterator[dict]:
        """Send a message and stream session events (blocking)."""
        q: Queue[dict | None] = Queue()

        async def _stream() -> None:
            async for event in self.send_async(agent_id, message, session_id=session_id):
                q.put(event)
            q.put(None)

        thread = Thread(target=lambda: asyncio.run(_stream()), daemon=True)
        thread.start()

        while True:
            event = q.get()
            if event is None:
                break
            yield event

        thread.join()

    async def send_async(
        self,
        agent_id: str,
        message: str,
        *,
        session_id: Optional[UUID] = None,
    ) -> AsyncIterator[dict]:
        """Send a message and stream session events (async)."""
        req = SendMessageRequest(
            agent_id=agent_id,
            message=message,
            tenant_id=self.tenant_id,
            session_id=session_id,
        )
        async with httpx.AsyncClient(base_url=self.url, timeout=None) as http:
            async with http.stream(
                "POST",
                "/sessions/send",
                json=req.model_dump(exclude_none=True, mode="json"),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        yield json.loads(line)
