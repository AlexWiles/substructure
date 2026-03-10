"""Send a message to an agent via the substructure2 runtime."""

import asyncio
import sys

from substructure2 import Client


async def main() -> None:
    message = " ".join(sys.argv[1:]) or "What's the weather in NYC?"

    client = Client("http://localhost:8080")
    resp = await client.send("weather", message)

    if resp.ok:
        print(f"Sent to session {resp.session_id}")
    else:
        print(f"Error: {resp.error}")


if __name__ == "__main__":
    asyncio.run(main())
