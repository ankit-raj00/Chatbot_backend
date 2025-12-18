from fastapi import HTTPException, status
from core.database import messages_collection, conversations_collection
from google import genai
from google.genai import types
from datetime import datetime
from bson import ObjectId
from utils.mcp_connection_manager import mcp_manager
import os
import json
import asyncio

# LangChain Imports
from langchain_core.messages import HumanMessage, AIMessage
from graph.builder import chat_graph

class ChatController:
    """Controller for chat operations with LangGraph + Gemini"""
    
    def __init__(self):
        # Initialize Gemini Client for File Uploads (Native)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.gemini_client = genai.Client(api_key=api_key)

    async def process_chat_stream(
        self, 
        user_id: str, 
        message: str, 
        conversation_id: str = None, 
        mcp_server_urls: list[str] = None, 
        model: str = "gemini-2.5-flash", 
        enabled_tools: list[str] = None,
        files: list = None
    ):
        """Process chat message with LangGraph streaming response"""
        from config.model_config import ModelConfig
        
        # 1. Validation & Setup
        if not ModelConfig.is_valid_model(model):
            model = ModelConfig.DEFAULT_MODEL
            
        if files and not ModelConfig.supports_images(model):
            yield f"data: {json.dumps({'error': 'Selected model does not support files'})}\n\n"
            return

        try:
            # 2. File Processing (Native Upload Logic - Unchanged)
            attachments = []
            files_content_parts = []
            
            if files:
                import tempfile
                from utils.cloudinary_handler import CloudinaryHandler
                cloudinary_handler = CloudinaryHandler()
                
                for file_obj in files:
                    tmp_path = None
                    try:
                        suffix = "." + file_obj.filename.split(".")[-1] if "." in file_obj.filename else ""
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            content = await file_obj.read()
                            tmp.write(content)
                            tmp_path = tmp.name
                        
                        # Upload to Cloudinary
                        cloudinary_url, cloudinary_public_id = await cloudinary_handler.upload_file(tmp_path)
                        
                        # Upload to Gemini Files API
                        gemini_file = self.gemini_client.files.upload(file=tmp_path)
                        
                        # Create LangChain compatible URI part
                        files_content_parts.append({
                            "type": "file",
                            "file_id": gemini_file.uri,
                            "mime_type": gemini_file.mime_type
                        })
                        
                        # Store Metadata
                        attachments.append({
                            "type": "file",
                            "original_name": file_obj.filename,
                            "mime_type": gemini_file.mime_type,
                            "size_bytes": gemini_file.size_bytes,
                            "cloudinary_url": cloudinary_url,
                            "cloudinary_public_id": cloudinary_public_id,
                            "gemini_uri": gemini_file.uri,
                            "gemini_name": gemini_file.name,
                            "gemini_uploaded_at": datetime.now()
                        })
                        
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            try: os.unlink(tmp_path) 
                            except: pass

            # 3. Conversation Management (Unchanged)
            if not conversation_id:
                new_conv = {
                    "user_id": user_id,
                    "title": message[:50],
                    "mcp_server_url": mcp_server_urls[0] if mcp_server_urls else None,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                result = await conversations_collection.insert_one(new_conv)
                conversation_id = str(result.inserted_id)
            else:
                await conversations_collection.update_one(
                    {"_id": ObjectId(conversation_id)},
                    {"$set": {"updated_at": datetime.now()}}
                )

            # 4. Save User Message (Unchanged)
            await messages_collection.insert_one({
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "user",
                "content": message,
                "attachments": attachments if attachments else None,
                "timestamp": datetime.now()
            })

            # 5. Connect to MCP Servers (New: Ensure connections for Graph)
            if mcp_server_urls:
                for url in mcp_server_urls:
                    await mcp_manager.connect(url)
            
            # 6. Build History (Native -> LangChain Message Objects)
            history_messages = []
            
            # Retrieve last 50 messages for context
            msgs_cursor = messages_collection.find({
                "conversation_id": conversation_id, 
                "user_id": user_id
            }).sort("timestamp", 1)
            
            stored_messages = await msgs_cursor.to_list(length=50) # Avoid excessive context
            
            for msg in stored_messages:
                 if str(msg.get("_id")) == str(conversation_id): continue # Skip conversation logic if mixed? No.
                 # Skip the current message we just inserted (logic match)
                 if msg["content"] == message and abs((msg["timestamp"] - datetime.now()).total_seconds()) < 1:
                     continue

                 content_parts = [{"type": "text", "text": msg.get("content", "")}]
                 
                 # Attachments
                 if msg.get("attachments"):
                     for att in msg.get("attachments"):
                         uri = att.get("gemini_uri") or att.get("uri")
                         if uri:
                             content_parts.append({
                                 "type": "file", 
                                 "file_id": uri, 
                                 "mime_type": att.get("mime_type")
                             })

                 if msg["role"] == "user":
                     history_messages.append(HumanMessage(content=content_parts))
                 else:
                     history_messages.append(AIMessage(content=msg.get("content", "")))

            # 7. Prepare Input
            current_message_content = [{"type": "text", "text": message}]
            current_message_content.extend(files_content_parts)
            
            input_message = HumanMessage(content=current_message_content)
            
            # 6.5. Fetch MCP Context (Resources & Prompts)
            try:
                available_resources = await mcp_manager.get_available_resources()
                available_prompts = await mcp_manager.get_available_prompts()
                
                system_context = ""
                
                # INJECT RESOURCES
                if available_resources:
                    system_context += "### Available MCP Context Resources\n"
                    system_context += "You can use the `read_mcp_resource` tool to read these if needed for context:\n"
                    for r in available_resources:
                        system_context += f"- **{r['name']}** ({r['mimeType']})\n  URI: `{r['uri']}`\n  Description: {r['description']}\n"
                    system_context += "\n"

                # INJECT PROMPTS
                if available_prompts:
                    system_context += "### Available MCP Prompts\n"
                    system_context += "These are standard prompt templates provided by the server. You can use them to guide your actions or structure your responses if relevant:\n"
                    for p in available_prompts:
                        args_str = ", ".join([f"{arg['name']}" for arg in p.get('arguments', [])])
                        system_context += f"- **{p['name']}**: {p['description']}\n  Arguments: {args_str}\n"
                    system_context += "\n"
                
                if system_context:
                    from langchain_core.messages import SystemMessage
                    # Prepend SystemMessage to history
                    # We add it as the VERY first message to act as a high-level instruction
                    history_messages.insert(0, SystemMessage(content=system_context))
                    
            except Exception as e:
                print(f"Failed to fetch MCP context: {e}")

            # Graph State
            graph_input = {
                "messages": history_messages + [input_message]
            }

            # 8. Run LangGraph & Stream to Client
            full_response_text = ""
            tool_steps = []
            
            # Pass enabled_tools and user_id via config
            config = {
                "configurable": {
                    "enabled_tools": enabled_tools or [],
                    "user_id": user_id,
                    "model": model
                }
            }
            
            async for event in chat_graph.astream_events(graph_input, version="v1", config=config):
                try:
                    if not isinstance(event, dict):
                         print(f"⚠️ Warning: Event is not a dict: {type(event)}")
                         continue
                         
                    event_type = event.get("event")
                    
                    # Stream Tokens
                    if event_type == "on_chat_model_stream":
                        data = event.get("data", {})
                        if not isinstance(data, dict):
                             # Fallback/Debug
                             print(f"⚠️ data is not dict: {type(data)} in event {event_type}")
                             continue

                        chunk = data.get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            content_str = ""
                            if isinstance(chunk.content, list):
                                # Ensure we only append text parts if it's a list
                                # or just stringify if appropriate, but usually list implies multimodal response parts.
                                # For simple chat stream, we probably just want the text.
                                for part in chunk.content:
                                    if isinstance(part, str):
                                        content_str += part
                                    elif isinstance(part, dict) and "text" in part:
                                        content_str += part["text"]
                            else:
                                content_str = str(chunk.content)
                                
                            full_response_text += content_str
                            yield f"data: {json.dumps({'chunk': content_str})}\n\n"
                    
                    # Tool Events
                    elif event_type == "on_tool_start":
                        tool_name = event.get("name")
                        data = event.get("data", {})
                        tool_args = data.get("input")
                        
                        yield f"data: {json.dumps({'status': f'Using tool: {tool_name}'})}\n\n"
                        yield f"data: {json.dumps({'tool_call': {'name': tool_name, 'args': tool_args}})}\n\n"
                        
                        tool_steps.append({
                            "name": tool_name,
                            "args": tool_args,
                            "status": "running"
                        })
                        
                    elif event_type == "on_tool_end":
                        tool_name = event.get("name")
                        data = event.get("data", {})
                        if not isinstance(data, dict):
                             print(f"⚠️ tool_end data is not dict: {type(data)}")
                             # Try to salvage if it's an object?
                             output = str(data)
                        else:
                             output = data.get("output")
                        
                        yield f"data: {json.dumps({'tool_output': {'name': tool_name, 'result': str(output)}})}\n\n"
                        
                        # Update tool steps? Simple append for now
                        if tool_steps and tool_steps[-1]["name"] == tool_name:
                             tool_steps[-1]["result"] = str(output)
                             tool_steps[-1]["status"] = "completed"
                        else:

                             # Fallback if ordering is weird (async)
                             pass
                             
                except Exception as loop_e:
                    print(f"Error processing event {event}: {loop_e}")
                    # Don't break the loop, try next event
                    continue

            # 9. Save Assistant Response (Unchanged Persistence)
            await messages_collection.insert_one({
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "model",
                "content": full_response_text,
                "tool_steps": tool_steps,
                "timestamp": datetime.now()
            })
            
            yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id})}\n\n"

        except Exception as e:
            print(f"Chat stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
