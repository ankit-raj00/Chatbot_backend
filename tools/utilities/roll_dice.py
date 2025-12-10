"""
Utility tool: Roll dice
"""
from tools.base import BaseTool
from typing import Any, Dict
import random


class RollDice(BaseTool):
    """Roll a dice"""
    
    @property
    def name(self) -> str:
        return "roll_dice"
    
    @property
    def description(self) -> str:
        return "Roll a dice with specified number of sides"
    
    @property
    def category(self) -> str:
        return "utilities"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sides": {
                    "type": "integer",
                    "description": "Number of sides on the dice (default: 6)",
                    "default": 6
                }
            }
        }
    
    async def execute(self, sides: int = 6) -> Dict[str, Any]:
        """Execute the tool"""
        result = random.randint(1, sides)
        return {"result": f"You rolled a {result} (out of {sides})"}
