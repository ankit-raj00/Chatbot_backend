"""
Utility tool: Get weather (mock implementation)
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import random

class GetWeatherArgs(BaseModel):
    location: str = Field(description="City name or location")

@tool(args_schema=GetWeatherArgs)
def get_weather(location: str):
    """Get current weather information for a location"""
    # Mock weather data
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
    temp = random.randint(15, 30)
    condition = random.choice(conditions)
    
    return {
        "result": f"Weather in {location}: {condition}, {temp}°C",
        "location": location,
        "temperature": temp,
        "condition": condition
    }
