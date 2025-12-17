"""
Utility tool: Roll dice
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import random

class RollDiceArgs(BaseModel):
    sides: int = Field(default=6, description="Number of sides on the dice (default: 6)")

@tool(args_schema=RollDiceArgs)
def roll_dice(sides: int = 6):
    """Roll a dice with specified number of sides"""
    result = random.randint(1, sides)
    return {"result": f"You rolled a {result} (out of {sides})"}
