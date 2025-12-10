from fastapi import HTTPException, status
from database import conversations_collection, messages_collection
from google import genai
from google.genai import types
try:
    from fastmcp import Client
except ImportError:
    Client = None
from contextlib import AsyncExitStack
from datetime import datetime
from bson import ObjectId
import os
import json

class ChatController:
    """Controller for chat operations with Gemini + MCP"""
    
    def __init__(self):
        # Initialize Gemini Client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            yield f"data: {json.dumps({'error': 'GEMINI_API_KEY not set'})}\n\n"
            return
        self.gemini_client = genai.Client(api_key=api_key)
        
        # Initialize MCP Client (Local) with absolute path
        import pathlib
        mcp_server_path = pathlib.Path(__file__).parent.parent / "mcp_server.py"
        self.local_mcp_client = Client(str(mcp_server_path))
    
    async def _handle_message_history(self, conversation_id: str, user_id: str, current_message: str = None, uploaded_files_data: list = None):
        """Construct conversation history with multimodal support"""
        # Get message history from MongoDB
        messages_cursor = messages_collection.find({
            "conversation_id": conversation_id,
            "user_id": user_id
        }).sort("timestamp", 1)
        messages_list = await messages_cursor.to_list(length=100)
        
        contents = []
        
        # Add history
        for msg in messages_list:
            role = "user" if msg["role"] == "user" else "model"
            parts = []
            
            # Add text
            if msg.get("content"):
                parts.append(types.Part.from_text(text=msg["content"]))
            
            # Add existing attachments (from DB)
            if msg.get("attachments"):
                for att in msg["attachments"]:
                    # Create file part from URI
                    parts.append(types.Part.from_uri(
                        file_uri=att["uri"],
                        mime_type=att["mime_type"]
                    ))
            
            if parts:
                contents.append(types.Content(role=role, parts=parts))

        # Add current message
        if current_message or uploaded_files_data:
            current_parts = []
            if current_message:
                current_parts.append(types.Part.from_text(text=current_message))
            
            if uploaded_files_data:
                for file_data in uploaded_files_data:
                    current_parts.append(types.Part.from_uri(
                        file_uri=file_data["uri"],
                        mime_type=file_data["mime_type"]
                    ))
            
            contents.append(types.Content(role="user", parts=current_parts))
            
        return contents

    async def process_chat_stream(self, user_id: str, message: str, conversation_id: str = None, mcp_server_url: str = None):
        """Process chat message with streaming response"""
        try:
            # Get or create conversation (Same as before)
            if not conversation_id:
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
                conv = await conversations_collection.find_one({"_id": ObjectId(conversation_id), "user_id": user_id})
                if not conv: raise HTTPException(status_code=404, detail="Conversation not found")
                await conversations_collection.update_one({"_id": ObjectId(conversation_id)}, {"$set": {"updated_at": datetime.now()}})

            # Save user message (Text only)
            user_message = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "user",
                "content": message,
                "timestamp": datetime.now()
            }
            await messages_collection.insert_one(user_message)

            # Build History (Unified Method)
            contents = await self._handle_message_history(conversation_id, user_id, current_message=message)

            # Call Gemini (Streaming)
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(self.local_mcp_client)
                tools = [self.local_mcp_client.session]

                if mcp_server_url:
                    try:
                        remote_client = Client(mcp_server_url)
                        await stack.enter_async_context(remote_client)
                        tools.append(remote_client.session)
                        print(f"Connected to remote MCP: {mcp_server_url}")
                    except Exception as e:
                        print(f"Failed to connect to remote MCP {mcp_server_url}: {e}")

                response_stream = self.gemini_client.aio.models.generate_content_stream(
                    model="gemini-2.0-flash-exp",
                    contents=contents,
                    config=types.GenerateContentConfig(tools=tools),
                )
                
                full_response = ""
                async for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        yield f"data: {json.dumps({'chunk': chunk.text})}\n\n"
                
                yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id})}\n\n"

            # Save assistant message
            assistant_message = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now()
            }
            await messages_collection.insert_one(assistant_message)
                
        except Exception as e:
            print(f"Chat error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def process_chat_stream_multimodal(self, user_id: str, message: str, conversation_id: str = None, mcp_server_urls: list = [], model: str = "gemini-2.5-flash", images: list = []):
        """Process multimodal chat message with streaming response using Files API"""
        import tempfile
        import shutil
        
        try:
            # Get or create conversation (Same as before)
            if not conversation_id:
                new_conv = {
                    "user_id": user_id,
                    "title": message[:50],
                    "mcp_server_urls": mcp_server_urls,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                result = await conversations_collection.insert_one(new_conv)
                conversation_id = str(result.inserted_id)
            else:
                conv = await conversations_collection.find_one({"_id": ObjectId(conversation_id), "user_id": user_id})
                if not conv: raise HTTPException(status_code=404, detail="Conversation not found")
                await conversations_collection.update_one({"_id": ObjectId(conversation_id)}, {"$set": {"updated_at": datetime.now()}})

            # 1. Upload Files to Gemini Files API
            uploaded_files_data = []
            if images:
                for file in images:
                    try:
                        # Create temp file
                        suffix = "." + file.filename.split(".")[-1] if "." in file.filename else ""
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            content = await file.read()
                            tmp.write(content)
                            tmp_path = tmp.name
                        
                        print(f"Uploading file: {file.filename} to Gemini Files API...")
                        # Upload to Gemini
                        gemini_file = self.gemini_client.files.upload(file=tmp_path)
                        print(f"Uploaded: {gemini_file.name}")
                        
                        uploaded_files_data.append({
                            "uri": gemini_file.uri,
                            "name": gemini_file.name, # resources/123
                            "mime_type": gemini_file.mime_type,
                            "original_name": file.filename
                        })
                        
                        # Cleanup temp file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        print(f"Failed to upload file {file.filename}: {e}")
            
            # Save user message with attachments
            user_message = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "user",
                "content": message,
                "attachments": uploaded_files_data, # Store metadata
                "timestamp": datetime.now()
            }
            await messages_collection.insert_one(user_message)

            # 2. Build History (Unified Method)
            contents = await self._handle_message_history(conversation_id, user_id, current_message=message, uploaded_files_data=uploaded_files_data)

            # 3. Call Gemini (Streaming)
            async with AsyncExitStack() as stack:
                # Local MCP
                try:
                    await stack.enter_async_context(self.local_mcp_client)
                    tools = [self.local_mcp_client.session]
                except Exception as e:
                    print(f"Warning: Local MCP: {e}")
                    tools = []

                # Remote MCPs
                if mcp_server_urls:
                    for url in mcp_server_urls:
                        try:
                            remote_client = Client(url)
                            await stack.enter_async_context(remote_client)
                            tools.append(remote_client.session)
                        except Exception as e:
                            print(f"Remote MCP Error {url}: {e}")

                response_stream = self.gemini_client.aio.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(tools=tools),
                )
                
                full_response = ""
                async for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        yield f"data: {json.dumps({'chunk': chunk.text})}\n\n"
                
                yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id})}\n\n"

            # Save assistant message
            assistant_message = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now()
            }
            await messages_collection.insert_one(assistant_message)

        except Exception as e:
            print(f"Multimodal Chat error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    # Keep the original non-streaming method for backward compatibility
    async def process_chat(self, user_id: str, message: str, conversation_id: str = None, mcp_server_url: str = None):
        """Process chat message with Gemini + MCP (non-streaming)"""
        # ... (keep existing implementation)
        pass
