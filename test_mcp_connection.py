
import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_connection():
    url = "https://first-mcp-server.fastmcp.app/mcp"
    print(f"Testing connection to: {url}")
    
    transport_config = {
        "url": url,
        "transport": "sse"
    }
    
    print(f"Config: {transport_config}")
    
    try:
        client = MultiServerMCPClient({
            "test_server": transport_config
        })
        
        print("Client initialized. Entering context...")
        async with client:
            print("Connected!")
            tools = await client.get_tools()
            print(f"Tools found: {len(tools)}")
            for t in tools:
                print(f" - {t.name}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())
