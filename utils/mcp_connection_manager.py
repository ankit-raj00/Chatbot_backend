from typing import Dict, Optional, List, Any
import asyncio
import time
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
        self._tool_cache: dict[str, tuple[list, float]] = {}   # url -> (tools, timestamp)
        self._tool_cache_ttl: int = 300                         # 5 minutes
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
            transport_config = {}
            
            if url.startswith("http://") or url.startswith("https://"):
                # Remote Server (using http transport as per user feedback)
                transport_config = {
                    "url": url,
                    "transport": "http"
                }
            elif url.endswith(".py"):
                 # Local Python Server (stdio)
                 # We assume key "command" is needed.
                 transport_config = {
                     "command": "python",
                     "args": [url],
                     "transport": "stdio"
                 }
            else:
                 # Fallback/Assumption: Local executable or unknown
                 # If it doesn't start with http, assume it's a command/path
                 transport_config = {
                     "command": url,
                     "args": [],
                     "transport": "stdio"
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
            self.invalidate_tool_cache(url)
            print(f"Successfully registered: {url}")
            return True
            
        except Exception as e:
            print(f"Failed to register MCP server {url}: {e}")
            return False
            
    async def get_all_langchain_tools(self) -> List[BaseTool]:
        """
        Get LangChain compatible tools from ALL registered clients.
        Tools are cached per-server for _tool_cache_ttl seconds.
        On cache miss: fetches from server. On error: serves stale cache.
        """
        all_tools = []
        now = time.monotonic()

        for url, client in self._clients.items():
            cached_tools, cached_at = self._tool_cache.get(url, (None, 0.0))
            cache_age = now - cached_at

            if cached_tools is not None and cache_age < self._tool_cache_ttl:
                # Cache HIT
                all_tools.extend(cached_tools)
                continue

            # Cache MISS — fetch from server
            try:
                tools = await client.get_tools()
                self._tool_cache[url] = (tools, now)
                all_tools.extend(tools)
                print(f"MCP tool cache refreshed for {url}: {len(tools)} tools")
            except Exception as e:
                print(f"Error fetching tools from {url}: {e}")
                if cached_tools is not None:
                    # Serve stale cache on error rather than failing
                    print(f"Serving stale tool cache for {url} ({cache_age:.0f}s old)")
                    all_tools.extend(cached_tools)

        return all_tools
        
    def invalidate_tool_cache(self, url: str = None) -> None:
        """
        Invalidate the tool cache. Call after connecting a new server
        or when the user adds/removes tools.
        url=None invalidates all servers.
        """
        if url:
            self._tool_cache.pop(url, None)
        else:
            self._tool_cache.clear()
        print(f"MCP tool cache invalidated for: {url or 'all servers'}")
        
    async def get_tools_from_server(self, url: str) -> List[BaseTool]:
        """
        Get LangChain compatible tools from A SPECIFIC registered client.
        """
        if url not in self._clients:
             raise ValueError(f"MCP server not connected: {url}")
             
        client = self._clients[url]
        try:
             return await client.get_tools()
        except Exception as e:
             print(f"Error fetching tools from {url}: {e}")
             raise e
        
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
        
    async def get_available_resources(self) -> List[Dict]:
        """Aggregate resources from all clients (Robust Metadata Version)"""
        all_resources = []
        for url, client in self._clients.items():
            try:
                # Use session context to access RAW MCP features (bypassing LangChain's buggy helpers)
                async with client.session(url) as session:
                    # session.list_resources() returns ListResourcesResult
                    result = await session.list_resources()
                    
                    if result and result.resources:
                        print(f"--- Fetched Resources from {url} ---")
                        for r in result.resources:
                            # Standard attributes on MCP Resource object
                            uri = r.uri
                            name = r.name
                            desc = r.description or "No description provided"
                            mime_type = r.mimeType or "application/octet-stream"
                            
                            # Debug Print for User
                            print(f"Resource found: {name} ({uri})")
                            print(f"  > Description: {desc}")
                            print(f"  > MimeType: {mime_type}")
                            
                            all_resources.append({
                                "uri": uri,
                                "name": name,
                                "description": desc,
                                "mimeType": mime_type,
                                "source_url": url
                            })
                        print("---------------------------------------")
            except Exception as e:
                print(f"Error fetching resources from {url}: {e}")
        return all_resources
        
    async def load_resource(self, uri: str) -> str:
        """Load a resource content"""
        # Try all clients to find who owns the URI
        # Optimally we would map URI schemes to servers, but simple iteration works for now.
        for url, client in self._clients.items():
            try:
                async with client.session(url) as session:
                    # Create ReadResourceRequest? or helper?
                    # session.read_resource(uri) usually works
                    result = await session.read_resource(uri)
                    # result.contents is list of TextResourceContents or BlobResourceContents
                    if result and result.contents:
                        # Concatenate contents? usually just one
                        content_str = ""
                        for c in result.contents:
                             if hasattr(c, "text") and c.text:
                                 content_str += c.text
                             elif hasattr(c, "blob") and c.blob:
                                 content_str += f"[Blob: {c.mimeType}]"
                        return content_str
            except Exception:
                # Not found on this server or error, try next
                continue
                
        raise ValueError(f"Resource not found: {uri}")

    async def get_available_prompts(self) -> List[Dict]:
        """Aggregate prompts from all clients"""
        all_prompts = []
        for url, client in self._clients.items():
            try:
                async with client.session(url) as session:
                    result = await session.list_prompts()
                    if result and result.prompts:
                        for p in result.prompts:
                            all_prompts.append({
                                "name": p.name,
                                "description": p.description,
                                "arguments": [
                                    {"name": arg.name, "description": arg.description, "required": arg.required}
                                    for arg in (p.arguments or [])
                                ],
                                "source_url": url
                            })
            except Exception as e:
                print(f"Error fetching prompts from {url}: {e}")
        return all_prompts
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any] = None) -> Any:
        """Get/Execute a prompt"""
        # We need to find the server that has this prompt.
        for url, client in self._clients.items():
            try:
                async with client.session(url) as session:
                     result = await session.get_prompt(name, arguments)
                     return result
            except Exception:
                continue
        raise ValueError(f"Prompt not found: {name}")

    # Note: Resource caching is handled by the client/session now.
    def get_cached_resources(self, url: str) -> List:
        return []

# Global instance
mcp_manager = MCPConnectionManager()
