"""
Base class for all native tools
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from google.genai import types


class BaseTool(ABC):
    """Base class for all native tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (used as function name in Gemini)"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass
    
    @property
    def category(self) -> str:
        """Tool category for grouping"""
        return "general"
    
    @property
    def requires_auth(self) -> bool:
        """Whether tool requires authentication"""
        return False
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        Tool parameters in JSON Schema format
        Example:
        {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
                "param2": {"type": "integer", "description": "..."}
            },
            "required": ["param1"]
        }
        """
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool
        Returns: Dict with result data
        """
        pass
    
    def to_gemini_function_declaration(self) -> types.FunctionDeclaration:
        """Convert tool to Gemini FunctionDeclaration"""
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters
        )
