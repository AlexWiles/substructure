"""Example substructure2 worker with a weather agent."""

import asyncio
import logging
import random

from substructure2 import Agent, Client, RunContext, Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

agent = Agent(
    "weather",
    model="stepfun/step-3.5-flash:free",
    instructions="You are a helpful weather assistant. Use the available tools to answer questions about weather.",
    llm_client="openrouter",
)


@agent.tool
async def get_weather(ctx: RunContext, city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name to look up.
    """
    temp = random.randint(50, 95)
    conditions = random.choice(["sunny", "cloudy", "rainy", "partly cloudy", "windy"])
    return f"{temp}°F and {conditions} in {city}"


@agent.tool
async def get_forecast(ctx: RunContext, city: str, days: int = 3) -> str:
    """Get a multi-day weather forecast.

    Args:
        city: The city name to look up.
        days: Number of days to forecast.
    """
    lines = []
    for i in range(days):
        temp = random.randint(50, 95)
        cond = random.choice(["sunny", "cloudy", "rainy", "partly cloudy"])
        lines.append(f"  Day {i + 1}: {temp}°F, {cond}")
    return f"Forecast for {city}:\n" + "\n".join(lines)


@agent.tool_plain
def convert_temperature(value: float, to_unit: str) -> str:
    """Convert a temperature between Fahrenheit and Celsius.

    Args:
        value: The temperature value.
        to_unit: Target unit, either 'celsius' or 'fahrenheit'.
    """
    if to_unit.lower().startswith("c"):
        converted = (value - 32) * 5 / 9
        return f"{value}°F = {converted:.1f}°C"
    else:
        converted = value * 9 / 5 + 32
        return f"{value}°C = {converted:.1f}°F"


async def main() -> None:
    client = Client("http://localhost:8080")
    worker = Worker(client, agents=[agent])

    # Start a session
    resp = await worker.send("weather", "What's the weather in NYC?")
    logging.info("Session started: %s", resp.session_id)

    # Run the worker loop
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
