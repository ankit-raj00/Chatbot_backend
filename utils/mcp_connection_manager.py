try:
    from fastmcp import Client
except ImportError:
    Client = Any  # Fallback for type hints if library missing

from typing import Dict, Optional, List, Any
import asyncio
from contextlib import AsyncExitStack


class MCPConnectionManager:
    """Manages persistent MCP server connections"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._connections: Dict[str, Client] = {}
        self._resources: Dict[str, List] = {}  # Cache resources per URL
        self._stack = AsyncExitStack()
        self._connection_locks: Dict[str, asyncio.Lock] = {}
        self._initialized = True
    
    async def connect(self, url: str) -> Optional[Client]:
        """
        Connect to an MCP server if not already connected.
        Returns the client session or None if connection fails.
        Also fetches and caches resources on first connection.
        """
        # Check if already connected
        if url in self._connections:
            print(f"Reusing existing connection to: {url}")
            return self._connections[url]
        
        # Get or create lock for this URL
        if url not in self._connection_locks:
            self._connection_locks[url] = asyncio.Lock()
        
        async with self._connection_locks[url]:
            # Double-check after acquiring lock
            if url in self._connections:
                return self._connections[url]
            
            try:
                print(f"Establishing new connection to MCP server: {url}")
                client = Client(url)
                await self._stack.enter_async_context(client)
                self._connections[url] = client
                print(f"Successfully connected to: {url}")
                
                # Fetch and cache resources
                await self._fetch_and_cache_resources(url, client)
                
                return client
            except Exception as e:
                print(f"Failed to connect to MCP server {url}: {e}")
                return None
    
    async def _fetch_and_cache_resources(self, url: str, client: Client):
        """Fetch resources from MCP server and cache them"""
        try:
            print(f"\n{'='*60}")
            print(f"Fetching resources from {url}...")
            print(f"{'='*60}")
            resources = await client.list_resources()
            
            if resources:
                resource_data = []
                for resource in resources:
                    try:
                        # Read resource content
                        content = await client.read_resource(resource.uri)
                        content_text = ""
                        if content and len(content) > 0:
                            content_text = content[0].text if hasattr(content[0], 'text') else str(content[0])
                        
                        resource_data.append({
                            'name': resource.name,
                            'uri': resource.uri,
                            'description': resource.description if hasattr(resource, 'description') else '',
                            'content': content_text[:5000]  # Limit to 5000 chars (increased from 1000)
                        })
                        
                        # Print detailed resource info
                        print(f"\n✓ Resource: {resource.name}")
                        print(f"  URI: {resource.uri}")
                        print(f"  Description: {resource.description if hasattr(resource, 'description') else 'N/A'}")
                        print(f"  Content Preview (first 1000 chars):")
                        print(f"  {content_text[:1000]}")
                        if len(content_text) > 1000:
                            print(f"  ... (truncated, total length: {len(content_text)} chars)")
                        
                    except Exception as e:
                        print(f"\n✗ Failed to read resource {resource.uri}: {e}")
                
                self._resources[url] = resource_data
                print(f"\n{'='*60}")
                print(f"✅ Successfully cached {len(resource_data)} resources from {url}")
                print(f"{'='*60}\n")
            else:
                self._resources[url] = []
                print(f"⚠️  No resources found on {url}\n")
        except Exception as e:
            print(f"❌ Failed to fetch resources from {url}: {e}\n")
            self._resources[url] = []
    
    def get_connection(self, url: str) -> Optional[Client]:
        """Get existing connection without attempting to connect"""
        return self._connections.get(url)
    
    def get_cached_resources(self, url: str) -> List:
        """Get cached resources for a URL"""
        return self._resources.get(url, [])
    
    async def disconnect(self, url: str):
        """Disconnect from a specific MCP server"""
        if url in self._connections:
            try:
                # The AsyncExitStack will handle cleanup
                del self._connections[url]
                if url in self._resources:
                    del self._resources[url]
                print(f"Disconnected from: {url}")
            except Exception as e:
                print(f"Error disconnecting from {url}: {e}")
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        print("Disconnecting from all MCP servers...")
        await self._stack.aclose()
        self._connections.clear()
        self._resources.clear()
        self._connection_locks.clear()
        print("All MCP connections closed")
    
    def get_active_connections(self) -> list[str]:
        """Get list of all active connection URLs"""
        return list(self._connections.keys())
    
    async def get_tools_for_urls(self, urls: list[str]) -> list:
        """
        Get Gemini Tool objects for the given URLs.
        Connects to servers that aren't already connected.
        Returns list of types.Tool objects.
        """
        from google.genai import types
        
        gemini_tools = []
        
        for url in urls:
            client = await self.connect(url)
            if client:
                try:
                    # List tools from the MCP server
                    mcp_tools = await client.list_tools()
                    
                    function_declarations = []
                    for tool in mcp_tools:
                        # Convert MCP tool to Gemini FunctionDeclaration
                        # MCP tool has: name, description, inputSchema
                        
                        # Ensure inputSchema is a dict
                        parameters = tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        
                        # Remove 'additionalProperties' if present as it might cause issues
                        if isinstance(parameters, dict):
                            if 'additionalProperties' in parameters:
                                del parameters['additionalProperties']
                        
                        func_decl = types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description or "",
                            parameters=parameters
                        )
                        function_declarations.append(func_decl)
                    
                    if function_declarations:
                        gemini_tools.append(types.Tool(function_declarations=function_declarations))
                        print(f"Converted {len(function_declarations)} tools from {url} to Gemini format")
                        
                except Exception as e:
                    print(f"Error fetching/converting tools from {url}: {e}")
                    
        return gemini_tools
    
    def get_resource_context_for_urls(self, urls: list[str]) -> str:
        """
        Get formatted resource context for the given URLs.
        Returns a string describing all available resources.
        """
        if not urls:
            return ""
        
        context_parts = []
        for url in urls:
            resources = self.get_cached_resources(url)
            if resources:
                context_parts.append(f"\n**MCP Server: {url}**")
                for resource in resources:
                    context_parts.append(f"- **{resource['name']}** (`{resource['uri']}`): {resource['description']}")
                    if resource['content']:
                        # Show full content (already limited to 5000 chars when cached)
                        context_parts.append(f"  Full Content:\n{resource['content']}")
        
        if context_parts:
            return "\n**Available MCP Resources:**\n" + "\n".join(context_parts)
        return ""

    async def call_tool_by_name(self, name: str, args: dict) -> Any:
        """
        Call a tool by name across all connected servers.
        Returns the result of the first server that successfully executes the tool.
        """
        for url, client in self._connections.items():
            try:
                # Check if tool exists on this client
                # FastMCP client doesn't have a simple "has_tool" check without listing
                # But we can try calling it.
                # Optimization: We could cache tool lists.
                
                # For now, list tools to check (or we could just try/catch)
                tools = await client.list_tools()
                if any(t.name == name for t in tools):
                    print(f"Executing tool {name} on {url}")
                    return await client.call_tool(name, arguments=args)
            except Exception as e:
                print(f"Error checking/calling tool {name} on {url}: {e}")
                continue
        
        raise ValueError(f"Tool {name} not found on any connected server")


# Global instance
mcp_manager = MCPConnectionManager()

