
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import sys

async def inspect_client():
    url = "https://first-mcp-server.fastmcp.app/mcp"
    client = MultiServerMCPClient({
        url: {
            "url": url,
            "transport": "http"
        }
    })
    
    print(f"Client Type: {type(client)}")
    print("Methods:")
    for method in dir(client):
        if not method.startswith("_"):
            print(f" - {method}")
            
    # Try calling get_resources if it exists
    if hasattr(client, "get_resources"):
        try:
            print("\nAttempting get_resources()...")
            res = await client.get_resources()
            print(f"Resources Found: {len(res)}")
            if len(res) > 0:
                first_res = res[0]
                print(f"Type: {type(first_res)}")
                print(f"Dir: {dir(first_res)}")
                # Check for standard fields
                print(f"Name: {getattr(first_res, 'name', 'N/A')}")
                print(f"URI: {getattr(first_res, 'uri', 'N/A')}")
                print(f"Metadata: {getattr(first_res, 'metadata', 'N/A')}")
                print(f"Description: {getattr(first_res, 'description', 'N/A')}")
        except Exception as e:
            print(f"get_resources failed: {e}")
            
if __name__ == "__main__":
    asyncio.run(inspect_client())
