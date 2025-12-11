"""
Tool Routes - API endpoints for native tools
"""
from fastapi import APIRouter
from controllers.tool_controller import ToolController

router = APIRouter(prefix="/api/tools", tags=["tools"])


@router.get("")
async def get_tools():
    """Get all available native tools"""
    return await ToolController.get_all_tools()


@router.get("/category/{category}")
async def get_tools_by_category(category: str):
    """Get tools by category"""
    return await ToolController.get_tools_by_category(category)
