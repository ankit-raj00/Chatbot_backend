"""
Utility tool: Get weather (mock implementation)
"""
from tools.base import BaseTool
from typing import Any, Dict
import random


class GetWeather(BaseTool):
    """Get weather information (mock)"""
    
    @property
    def name(self) -> str:
        return "get_weather"
    
    @property
    def description(self) -> str:
        return "Get current weather information for a location"
    
    @property
    def category(self) -> str:
        return "utilities"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                }
            },
            "required": ["location"]
        }
    
    async def execute(self, location: str) -> Dict[str, Any]:
        """Execute the tool (mock implementation)"""
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
