"""Send a message to an agent and stream events."""

import json
import sys

from substructure2 import Client

message = " ".join(sys.argv[1:]) or "What's the weather in NYC?"

for event in Client("http://localhost:8080").send("weather", message):
    event_type = event.get("payload", {}).get("type", "?")
    print(f"{event_type}: {json.dumps(event, indent=2)}")
