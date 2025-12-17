from typing import Dict, Optional, List, Any
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

class MCPConnectionManager:
    """
    Manages MCP server connections using LangChain's MultiServerMCPClient.
    Strictly uses langchain-mcp-adapters without direct SDK usage or fastmcp.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        # Store clients: Map URL -> MultiServerMCPClient
        self._clients: Dict[str, MultiServerMCPClient] = {}
        self._initialized = True
    
    async def connect(self, url: str) -> bool:
        """
        Register an MCP server via MultiServerMCPClient.
        We create a separate client for each URL to support dynamic addition.
        """
        if url in self._clients:
            print(f"MCP Server already registered: {url}")
            return True
            
        try:
            print(f"Registering MCP Client for: {url}")
            
            # Determine transport config
            # For now, we assume remote URLs are SSE. 
            # If we needed stdio, we'd need a different config structure.
            transport_config = {
                "url": url,
                "transport": "sse" 
            }
            
            # Initialize Client
            # Note: We use the URL itself as the server name key
            client = MultiServerMCPClient({
                url: transport_config
            })
            
            # Validate connection by trying to fetch tools immediately?
            # The client might be lazy. Let's try to get tools to fail fast.
            # await client.get_tools() 
            # Commented out to avoid immediate overhead/failure if lazy is desired.
            
            self._clients[url] = client
            print(f"Successfully registered: {url}")
            return True
            
        except Exception as e:
            print(f"Failed to register MCP server {url}: {e}")
            return False
            
    async def get_all_langchain_tools(self) -> List[BaseTool]:
        """
        Get LangChain compatible tools from ALL registered clients.
        Delegates to MultiServerMCPClient.get_tools().
        """
        all_tools = []
        for url, client in self._clients.items():
            try:
                # get_tools() returns list of BaseTool (LangChain native)
                tools = await client.get_tools()
                all_tools.extend(tools)
            except Exception as e:
                print(f"Error fetching tools from {url}: {e}")
                
        return all_tools
        
    async def call_tool_by_name(self, name: str, args: dict) -> Any:
        """
        Finds the tool in any registered client and executes it.
        Uses the LangChain tool's own ainvoke method.
        """
        # 1. Get all tools (expensive? maybe cache tools later)
        tools = await self.get_all_langchain_tools()
        
        # 2. Find tool
        target_tool = next((t for t in tools if t.name == name), None)
        
        if not target_tool:
            raise ValueError(f"Tool '{name}' not found on any connected server.")
            
        # 3. Execute
        # LangChain tools support .ainvoke(args)
        print(f"Executing MCP tool via Adapter: {name}")
        return await target_tool.ainvoke(args)
        
    # --- Legacy Methods (Stubs/Depracated) ---
    
    def get_active_connections(self) -> list[str]:
        return list(self._clients.keys())

    async def disconnect(self, url: str):
        if url in self._clients:
            del self._clients[url]
            print(f"Unregistered: {url}")

    async def disconnect_all(self):
        self._clients.clear()
        
    # Note: Resource caching is removed as MultiServerMCPClient abstracts the session.
    def get_cached_resources(self, url: str) -> List:
        return []

# Global instance
mcp_manager = MCPConnectionManager()
