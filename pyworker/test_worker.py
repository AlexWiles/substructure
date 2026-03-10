"""End-to-end test: Python worker echoes messages, client verifies."""

import asyncio
import logging

from substructure import Substructure, Session, AgentContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def main():
    client = Substructure("localhost:50051")

    # --- Worker ---
    worker = client.worker("test-py-worker")
    echo = worker.agent("echo")

    @echo.on_user_message
    async def on_user_message(
        session: Session, ctx: AgentContext, message: str, stream: bool,
    ):
        logging.info("worker got: %s", message)
        return session.decide(
            actions=[session.done(text=f"echo: {message}")],
            state={"messages": [message]},
        )

    # Start worker as a background task
    worker_task = asyncio.create_task(worker.start())

    # --- Client (runs a session) ---
    print("Running session...")
    async for event in client.run_session(agent="echo", message="Hello from Python!"):
        print(f"  [{event.event_type}] seq={event.sequence}")
        if event.is_done:
            artifacts = event.payload.get("artifacts", [])
            for a in artifacts:
                for part in a.get("parts", []):
                    if "text" in part:
                        print(f"  Result: {part['text']}")
            break

    print("Done!")
    worker_task.cancel()
    await client.close()


asyncio.run(main())
