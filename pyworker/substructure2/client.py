"""Client for sending messages to a substructure2 runtime."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

import httpx

from .types import SendMessageRequest, SendMessageResponse


class Client:
    """Send messages to agents on a substructure2 runtime.

    Usage::

        client = Client("http://localhost:8080")
        resp = await client.send("weather", "What's the weather in NYC?")
        print(resp.session_id)
    """

    def __init__(self, url: str, *, tenant_id: str = "default") -> None:
        self.url = url.rstrip("/")
        self.tenant_id = tenant_id

    async def send(
        self,
        agent_id: str,
        message: str,
        *,
        session_id: Optional[UUID] = None,
    ) -> SendMessageResponse:
        """Send a message to an agent, creating a session if needed."""
        req = SendMessageRequest(
            agent_id=agent_id,
            message=message,
            tenant_id=self.tenant_id,
            session_id=session_id,
        )
        async with httpx.AsyncClient(base_url=self.url, timeout=30.0) as http:
            resp = await http.post(
                "/sessions/send",
                json=req.model_dump(exclude_none=True, mode="json"),
            )
            resp.raise_for_status()
            return SendMessageResponse.model_validate(resp.json())
