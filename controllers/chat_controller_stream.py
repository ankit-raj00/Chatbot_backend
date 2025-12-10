from fastapi import HTTPException, status
from database import conversations_collection, messages_collection
from google import genai
from google.genai import types
from fastmcp import Client
from contextlib import AsyncExitStack
from datetime import datetime
from bson import ObjectId
import os
import json

class ChatController:
    """Controller for chat operations with Gemini + MCP"""
    
    def __init__(self):
        # Initialize Gemini Client
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBU-4y8LmHJwzxOZNHsAdbvidILliJfZQo")
        self.gemini_client = genai.Client(api_key=api_key)
        
        # Initialize MCP Client (Local) with absolute path
        import pathlib
        mcp_server_path = pathlib.Path(__file__).parent.parent / "mcp_server.py"
        self.local_mcp_client = Client(str(mcp_server_path))
    
    async def process_chat_stream(self, user_id: str, message: str, conversation_id: str = None, mcp_server_url: str = None):
        """Process chat message with streaming response"""
        try:
            # Get or create conversation
            if not conversation_id:
                # Create new conversation
                new_conv = {
                    "user_id": user_id,
                    "title": message[:50],
                    "mcp_server_url": mcp_server_url,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                result = await conversations_collection.insert_one(new_conv)
                conversation_id = str(result.inserted_id)
            else:
                # Verify conversation belongs to user
                conv = await conversations_collection.find_one({
                    "_id": ObjectId(conversation_id),
                    "user_id": user_id
                })
                if not conv:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Conversation not found"
                    )
                
                # Update conversation timestamp
                await conversations_collection.update_one(
                    {"_id": ObjectId(conversation_id)},
                    {"$set": {"updated_at": datetime.now()}}
                )

            # Get message history from MongoDB
            messages_cursor = messages_collection.find({
                "conversation_id": conversation_id,
                "user_id": user_id
            }).sort("timestamp", 1)
            messages_list = await messages_cursor.to_list(length=100)
            
            history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages_list
            ]

            # Save user message
            user_message = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "user",
                "content": message,
                "timestamp": datetime.now()
            }
            await messages_collection.insert_one(user_message)

            # Convert history to Gemini format
            contents = []
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
            
            # Call Gemini with MCP tools and resources (streaming)
            async with AsyncExitStack() as stack:
                # Always use local tools
                await stack.enter_async_context(self.local_mcp_client)
                tools = [self.local_mcp_client.session]

                # If remote URL provided, connect to it
                if mcp_server_url:
                    try:
                        remote_client = Client(mcp_server_url)
                        await stack.enter_async_context(remote_client)
                        tools.append(remote_client.session)
                        print(f"Connected to remote MCP: {mcp_server_url}")
                        
                        # Fetch and include resources from remote server
                        try:
                            print(f"Attempting to list resources from {mcp_server_url}")
                            resources = await remote_client.list_resources()
                            print(f"Found {len(resources) if resources else 0} resources")
                            if resources:
                                resource_context = "\n\n**Available Resources:**\n"
                                for resource in resources:
                                    print(f"Processing resource: {resource.name} - {resource.uri}")
                                    try:
                                        resource_content = await remote_client.read_resource(resource.uri)
                                        print(f"Resource content type: {type(resource_content)}, length: {len(resource_content) if resource_content else 0}")
                                        resource_context += f"\n- {resource.name} ({resource.uri}): {resource.description}\n"
                                        if resource_content and len(resource_content) > 0:
                                            # Add resource content to context
                                            content_text = resource_content[0].text if hasattr(resource_content[0], 'text') else str(resource_content[0])
                                            print(f"Resource content preview: {content_text[:100]}")
                                            resource_context += f"  Content: {content_text[:500]}...\n"  # Limit to 500 chars
                                    except Exception as e:
                                        print(f"Failed to read resource {resource.uri}: {e}")
                                
                                # Add resource context as a system message
                                if resource_context:
                                    print(f"Adding resource context to conversation: {resource_context[:200]}")
                                    contents.insert(0, types.Content(
                                        role="user",
                                        parts=[types.Part.from_text(text=resource_context)]
                                    ))
                        except Exception as e:
                            print(f"Failed to fetch resources: {e}")
                    except Exception as e:
                        print(f"Failed to connect to remote MCP: {e}")
            
                # Add current user message
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

                # Use streaming API
                response_stream = self.gemini_client.aio.models.generate_content_stream(
                    model="gemini-2.0-flash-exp",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=tools,
                    ),
                )
                
                # Collect full response while streaming
                full_response = ""
                async for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        # Yield SSE formatted chunk
                        yield f"data: {json.dumps({'chunk': chunk.text})}\n\n"
                
                # Send conversation_id in final message
                yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id})}\n\n"

            # Save assistant message to database
            assistant_message = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now()
            }
            await messages_collection.insert_one(assistant_message)
                
        except HTTPException:
            raise
        except Exception as e:
            print(f"Chat error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    # Keep the original non-streaming method for backward compatibility
    async def process_chat(self, user_id: str, message: str, conversation_id: str = None, mcp_server_url: str = None):
        """Process chat message with Gemini + MCP (non-streaming)"""
        # ... (keep existing implementation)
        pass
