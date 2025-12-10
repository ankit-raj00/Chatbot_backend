from fastmcp import FastMCP
import random

mcp = FastMCP(name="Local Demo Tools")

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll n_dice 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

@mcp.tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    # Mock implementation
    weathers = ["Sunny", "Rainy", "Cloudy", "Snowy"]
    temps = range(10, 30)
    return f"The weather in {city} is {random.choice(weathers)} with a temperature of {random.choice(temps)}°C."

if __name__ == "__main__":
    mcp.run()
