from fastapi import HTTPException, status
from core.database import messages_collection, conversations_collection
from google import genai
from google.genai import types
try:
    from fastmcp import Client
except ImportError:
    Client = None
from contextlib import AsyncExitStack
from datetime import datetime
from bson import ObjectId
from utils.mcp_connection_manager import mcp_manager
from tools import get_tool, execute_tool
import os

class ChatController:
    """Controller for chat operations with Gemini + MCP"""
    
    def __init__(self):
        # Initialize Gemini Client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.gemini_client = genai.Client(api_key=api_key)
        
        # MCP connections are now managed by the global mcp_manager
        # Local MCP server path for reference
        import pathlib
        self.local_mcp_path = str(pathlib.Path(__file__).parent.parent / "mcp_server.py")
    
    
    async def process_chat(self, user_id: str, message: str, conversation_id: str = None, mcp_server_url: str = None):
        """Process chat message with Gemini + MCP"""
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
            
            # Call Gemini with MCP tools and resources
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

                response = await self.gemini_client.aio.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=tools,
                    ),
                )
                
                print(f"Gemini Response: {response}")
                print(f"Gemini Response Text: {response.text}")
                
                assistant_response = response.text

            # Save assistant message
            assistant_message = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now()
            }
            await messages_collection.insert_one(assistant_message)

            return {
                "response": assistant_response,
                "conversation_id": conversation_id
            }
                
        except HTTPException:
            raise
        except Exception as e:
            print(f"Chat error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    async def process_chat_stream(self, user_id: str, message: str, conversation_id: str = None, mcp_server_urls: list[str] = None, model: str = "gemini-2.5-flash", enabled_tools: list[str] = None):
        """Process chat message with streaming response"""
        import json
        from config.model_config import ModelConfig
        
        # Validate model
        if not ModelConfig.is_valid_model(model):
            model = ModelConfig.DEFAULT_MODEL

        try:
            # Get or create conversation
            if not conversation_id:
                new_conv = {
                    "user_id": user_id,
                    "title": message[:50],
                    "mcp_server_url": mcp_server_urls[0] if mcp_server_urls and len(mcp_server_urls) > 0 else None,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                result = await conversations_collection.insert_one(new_conv)
                conversation_id = str(result.inserted_id)
            else:
                conv = await conversations_collection.find_one({
                    "_id": ObjectId(conversation_id),
                    "user_id": user_id
                })
                if not conv:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
                await conversations_collection.update_one(
                    {"_id": ObjectId(conversation_id)},
                    {"$set": {"updated_at": datetime.now()}}
                )

            # Get message history
            messages_cursor = messages_collection.find({
                "conversation_id": conversation_id,
                "user_id": user_id
            }).sort("timestamp", 1)
            messages_list = await messages_cursor.to_list(length=100)
            
            # Save user message
            await messages_collection.insert_one({
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "user",
                "content": message,
                "timestamp": datetime.now()
            })

            # Convert history to Gemini format with auto re-upload for expired files
            from utils.file_expiry import is_gemini_file_expired
            from utils.cloudinary_handler import CloudinaryHandler
            
            contents = []
            cloudinary_handler = CloudinaryHandler()
            files_to_update = []  # Track files that need DB updates
            
            for msg in messages_list:
                role = "user" if msg["role"] == "user" else "model"
                parts = []
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=msg["content"]))
                
                # Add attachments if present
                if "attachments" in msg and msg["attachments"]:
                    for attachment in msg["attachments"]:
                        try:
                            # Check if Gemini file has expired
                            gemini_uploaded_at = attachment.get("gemini_uploaded_at")
                            gemini_uri = attachment.get("gemini_uri") or attachment.get("uri")
                            cloudinary_url = attachment.get("cloudinary_url")
                            
                            # If file expired and we have Cloudinary backup, re-upload
                            if gemini_uploaded_at and is_gemini_file_expired(gemini_uploaded_at) and cloudinary_url:
                                print(f"⚠️ Gemini file expired for {attachment.get('original_name')}. Re-uploading from Cloudinary...")
                                
                                # Download from Cloudinary
                                tmp_path = cloudinary_handler.download_file(cloudinary_url)
                                
                                try:
                                    # Re-upload to Gemini Files API
                                    gemini_file = self.gemini_client.files.upload(file=tmp_path)
                                    print(f"✅ Re-uploaded to Gemini: {gemini_file.uri}")
                                    
                                    # Update attachment metadata
                                    attachment["gemini_uri"] = gemini_file.uri
                                    attachment["gemini_name"] = gemini_file.name
                                    attachment["gemini_uploaded_at"] = datetime.now()
                                    
                                    # Track for DB update
                                    files_to_update.append({
                                        "message_id": msg["_id"],
                                        "attachment": attachment
                                    })
                                    
                                    gemini_uri = gemini_file.uri
                                finally:
                                    # Cleanup temp file
                                    if os.path.exists(tmp_path):
                                        os.unlink(tmp_path)
                            
                            # Add file to parts
                            if gemini_uri:
                                parts.append(types.Part.from_uri(
                                    file_uri=gemini_uri,
                                    mime_type=attachment["mime_type"]
                                ))
                            elif "data" in attachment:
                                # Fallback for older messages with base64
                                import base64
                                data_bytes = base64.b64decode(attachment["data"])
                                parts.append(types.Part.from_bytes(
                                    data=data_bytes,
                                    mime_type=attachment["mime_type"]
                                ))
                        except Exception as e:
                            print(f"Failed to add attachment to history: {e}")
                
                if parts:
                    contents.append(types.Content(role=role, parts=parts))
            
            # Update MongoDB with new Gemini URIs for re-uploaded files
            for update_info in files_to_update:
                try:
                    await messages_collection.update_one(
                        {"_id": update_info["message_id"]},
                        {"$set": {"attachments": [update_info["attachment"]]}}
                    )
                except Exception as e:
                    print(f"Failed to update message with new Gemini URI: {e}")
            
            
            # Call Gemini with streaming and manual tool handling
            async with AsyncExitStack() as stack:
                tools = []
                
                # Add enabled native tools
                if enabled_tools and len(enabled_tools) > 0:
                    native_tool_declarations = []
                    for tool_id in enabled_tools:
                        tool = get_tool(tool_id)
                        if tool:
                            native_tool_declarations.append(tool.to_gemini_function_declaration())
                    
                    if native_tool_declarations:
                        tools.append(types.Tool(function_declarations=native_tool_declarations))
                        print(f"[DEBUG] Added {len(native_tool_declarations)} native tools")
                
                # Only connect to MCP if user has selected servers
                if mcp_server_urls and len(mcp_server_urls) > 0:
                    # Use connection manager to get/create persistent connections
                    mcp_tools = await mcp_manager.get_tools_for_urls(mcp_server_urls)
                    if mcp_tools:
                        tools.extend(mcp_tools)
                        print(f"[DEBUG] Added {len(mcp_tools)} MCP tools")
                    
                    # Get cached resource context
                    resource_context = mcp_manager.get_resource_context_for_urls(mcp_server_urls)
            
                # Add system instruction if tools are available
                if tools:
                    system_instruction = (
                        "You have access to tools that you can use when appropriate. "
                        "However, you should answer questions directly when tools are not needed. "
                        "For example, if asked to write an essay or explain a concept, answer directly. "
                        "Only use tools when they are specifically relevant to the user's request."
                    )
                    
                    # Add resource context if available
                    if mcp_server_urls and len(mcp_server_urls) > 0:
                        resource_context = mcp_manager.get_resource_context_for_urls(mcp_server_urls)
                        if resource_context:
                            system_instruction += "\n\n" + resource_context
                    
                    contents.insert(0, types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=system_instruction)]
                    ))
                    contents.insert(1, types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="Understood. I will use tools only when appropriate and answer general questions directly.")]
                    ))
            
                # Add current user message
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

                # Pass tools if available - Gemini will use them when appropriate
                config_params = {}
                if tools:
                    config_params["tools"] = tools
                    print(f"[DEBUG] Sending {len(tools)} tool(s) to Gemini")
                    for tool in tools:
                        print(f"[DEBUG] Tool: {tool.function_declarations[0].name if hasattr(tool, 'function_declarations') else 'unknown'}")
                    # Disable automatic function calling to handle injection manually
                    # config_params["automatic_function_calling"] = False # Default is False usually

                # Tool execution loop
                while True:
                    response_stream = await self.gemini_client.aio.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(**config_params),
                    )
                    
                    full_response = ""
                    function_calls = []
                    
                    async for chunk in response_stream:
                        # Check for function calls
                        if chunk.function_calls:
                            for fc in chunk.function_calls:
                                function_calls.append(fc)
                        
                        if chunk.text:
                            full_response += chunk.text
                            yield f"data: {json.dumps({'chunk': chunk.text})}\n\n"
                    
                    # If no function calls, we are done
                    if not function_calls:
                        break
                    
                    # Handle function calls
                    # Add model's function call message to history
                    parts = []
                    for fc in function_calls:
                        parts.append(types.Part.from_function_call(name=fc.name, args=fc.args))
                    contents.append(types.Content(role="model", parts=parts))
                    
                    # Execute tools and add responses
                    response_parts = []
                    for fc in function_calls:
                        print(f"Executing tool: {fc.name}")
                        yield f"data: {json.dumps({'status': f'Using tool: {fc.name}'})}\n\n"
                        
                        args = fc.args
                        # Inject user_id for Google Drive tools
                        if fc.name in ["list_google_drive_folders", "create_google_drive_folder"]:
                            print(f"Injecting user_id for {fc.name}")
                            args["user_id"] = user_id
                        try:
                            # 1. Native Tools (Google Drive, Utilities)
                            # Native tools are executed via tools.execute_tool
                            from tools import execute_tool, get_tool
                            
                            is_native_tool = get_tool(fc.name) is not None
                            
                            if is_native_tool:
                                print(f"[DEBUG] Executing native tool: {fc.name}")
                                
                                # Inject user_id for Google Drive tools
                                if fc.name in ["list_google_drive_folders", "create_google_drive_folder"]:
                                    print(f"[DEBUG] Injecting user_id: {user_id}")
                                    args["user_id"] = user_id
                                
                                # Execute
                                result = await execute_tool(fc.name, **args)
                                
                                # Format response
                                if isinstance(result, dict) and "result" in result:
                                    response_content = result["result"]
                                else:
                                    response_content = result
                                    
                            else:
                                # 2. MCP Tools
                                # Executed via mcp_manager
                                print(f"[DEBUG] Executing MCP tool: {fc.name}")
                                result = await mcp_manager.call_tool_by_name(fc.name, args)
                                response_content = result if isinstance(result, str) else str(result)
                                if isinstance(result, list) and len(result) > 0 and hasattr(result[0], 'text'):
                                    response_content = result[0].text
                            
                            print(f"[DEBUG] Tool output: {response_content}")
                            
                            response_parts.append(types.Part.from_function_response(
                                name=fc.name,
                                response={"result": response_content}
                            ))
                        except Exception as e:
                            print(f"Tool execution failed: {e}")
                            response_parts.append(types.Part.from_function_response(
                                name=fc.name,
                                response={"error": str(e)}
                            ))
                            
                    contents.append(types.Content(role="user", parts=response_parts))
                    # Loop continues to generate response based on tool output
                
                yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id})}\n\n"

            # Save assistant message
            await messages_collection.insert_one({
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now()
            })
                
        except Exception as e:
            print(f"Chat stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    

    async def process_chat_stream_multimodal(
        self, 
        user_id: str, 
        message: str, 
        conversation_id: str = None, 
        mcp_server_urls: list[str] = None,
        model: str = "gemini-2.5-flash",
        images: list = None,
        enabled_tools: list[str] = None
    ):
        """Process multimodal chat message with streaming response"""
        import json
        from config.model_config import ModelConfig
        from utils.file_handler import FileHandler
        
        print(f"\n{'='*50}")
        print(f"MULTIMODAL REQUEST RECEIVED")
        print(f"User ID: {user_id}")
        print(f"Message: {message}")
        print(f"Model: {model}")
        print(f"Images count: {len(images) if images else 0}")
        print(f"{'='*50}\n")
        try:
            # Validate model
            if not ModelConfig.is_valid_model(model):
                model = ModelConfig.DEFAULT_MODEL
            
            # Check if model supports images
            if images and not ModelConfig.supports_images(model):
                yield f"data: {json.dumps({'error': 'Selected model does not support images'})}\n\n"
                return
            
            
            # Process images/files with dual upload (Cloudinary + Gemini)
            user_parts = [types.Part.from_text(text=message)]
            attachments = []
            
            if images:
                import tempfile
                from utils.cloudinary_handler import CloudinaryHandler
                
                cloudinary_handler = CloudinaryHandler()
                
                for image in images:
                    tmp_path = None
                    try:
                        # Save to temp file
                        suffix = "." + image.filename.split(".")[-1] if "." in image.filename else ""
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            content = await image.read()
                            tmp.write(content)
                            tmp_path = tmp.name
                        
                        # 1. Upload to Cloudinary (permanent storage)
                        print(f"Uploading {image.filename} to Cloudinary...")
                        cloudinary_url, cloudinary_public_id = cloudinary_handler.upload_file(tmp_path)
                        print(f"Cloudinary URL: {cloudinary_url}")
                        
                        # 2. Upload to Gemini Files API (48h context)
                        print(f"Uploading {image.filename} to Gemini Files API...")
                        gemini_file = self.gemini_client.files.upload(file=tmp_path)
                        print(f"Gemini URI: {gemini_file.uri}")
                        
                        # Add to parts for current message
                        user_parts.append(types.Part.from_uri(
                            file_uri=gemini_file.uri,
                            mime_type=gemini_file.mime_type
                        ))
                        
                        # Store metadata for DB (both Cloudinary and Gemini info)
                        attachments.append({
                            "type": "file",
                            "original_name": image.filename,
                            "mime_type": gemini_file.mime_type,
                            "size_bytes": gemini_file.size_bytes,
                            "cloudinary_url": cloudinary_url,
                            "cloudinary_public_id": cloudinary_public_id,
                            "gemini_uri": gemini_file.uri,
                            "gemini_name": gemini_file.name,
                            "gemini_uploaded_at": datetime.now()
                        })
                        
                    except Exception as e:
                        print(f"Failed to process file {image.filename}: {e}")
                    finally:
                        # Cleanup temp file
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
            
            # Get or create conversation
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
                conv = await conversations_collection.find_one({
                    "_id": ObjectId(conversation_id),
                    "user_id": user_id
                })
                if not conv:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
                await conversations_collection.update_one(
                    {"_id": ObjectId(conversation_id)},
                    {"$set": {"updated_at": datetime.now()}}
                )

            # Get message history
            messages_cursor = messages_collection.find({
                "conversation_id": conversation_id,
                "user_id": user_id
            }).sort("timestamp", 1)
            messages_list = await messages_cursor.to_list(length=100)
            
            # Save user message with attachments
            await messages_collection.insert_one({
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "user",
                "content": message,
                "attachments": attachments if attachments else None,
                "timestamp": datetime.now()
            })

            # Convert history to Gemini format (including current message implicitly if we re-fetch, but we'll append current manually)
            # Actually, we should include previous messages + current message
            
            # Convert history to Gemini format
            contents = []
            for msg in messages_list:
                role = "user" if msg["role"] == "user" else "model"
                parts = []
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=msg["content"]))
                
                # Add attachments if present (Files API context)
                if "attachments" in msg and msg["attachments"]:
                    for att in msg["attachments"]:
                        try:
                            # Support both 'uri' (new) and 'gemini_uri' (old)
                            file_uri = att.get("uri") or att.get("gemini_uri")
                            if file_uri:
                                parts.append(types.Part.from_uri(
                                    file_uri=file_uri,
                                    mime_type=att["mime_type"]
                                ))
                        except Exception as e:
                            print(f"Failed to add attachment to history: {e}")
                
                if parts:
                    contents.append(types.Content(role=role, parts=parts))
            
            
            # Call Gemini with streaming and multimodal support
            async with AsyncExitStack() as stack:
                tools = []
                
                # Only connect to MCP if user has selected servers
                if mcp_server_urls and len(mcp_server_urls) > 0:
                    # Use connection manager to get/create persistent connections
                    tools = await mcp_manager.get_tools_for_urls(mcp_server_urls)
                    
                    # Get cached resource context
                    resource_context = mcp_manager.get_resource_context_for_urls(mcp_server_urls)
            
                # Build multimodal content for CURRENT message
                current_parts = [types.Part.from_text(text=message)]
                
                # Add system instruction if tools are available
                if tools:
                    system_instruction = (
                        "You have access to tools that you can use when appropriate. "
                        "However, you should answer questions directly when tools are not needed. "
                        "For example, if asked to write an essay or explain a concept, answer directly. "
                        "Only use tools when they are specifically relevant to the user's request."
                    )
                    
                    # Add resource context if available
                    if resource_context:
                        system_instruction += "\n\n" + resource_context
                    
                    contents.insert(0, types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=system_instruction)]
                    ))
                    contents.insert(1, types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="Understood. I will use tools only when appropriate and answer general questions directly.")]
                    ))
            
                # Build multimodal content for CURRENT message (continued)
                
                # Add current attachments
                if user_parts: # user_parts was populated with file URIs earlier
                     # Filter out the text part we just added to user_parts in previous step?
                     # Wait, user_parts in previous block was: [text, uri1, uri2...]
                     # But we just reconstructed history from DB which DOES NOT include the current message yet (messages_list was fetched BEFORE insert)
                     # So we need to append the current message here.
                     pass

                # Actually, let's use the user_parts we built earlier
                contents.append(types.Content(role="user", parts=user_parts))

                # Pass tools if available - Gemini will use them when appropriate
                config_params = {}
                if tools:
                    config_params["tools"] = tools

                # Tool execution loop
                while True:
                    response_stream = await self.gemini_client.aio.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(**config_params),
                    )
                    
                    full_response = ""
                    function_calls = []
                    
                    async for chunk in response_stream:
                        # Check for function calls
                        if chunk.function_calls:
                            for fc in chunk.function_calls:
                                function_calls.append(fc)
                        
                        if chunk.text:
                            full_response += chunk.text
                            yield f"data: {json.dumps({'chunk': chunk.text})}\n\n"
                    
                    # If no function calls, we are done
                    if not function_calls:
                        break
                    
                    # Handle function calls
                    parts = []
                    for fc in function_calls:
                        parts.append(types.Part.from_function_call(name=fc.name, args=fc.args))
                    contents.append(types.Content(role="model", parts=parts))
                    
                    # Execute tools and add responses
                    response_parts = []
                    for fc in function_calls:
                        print(f"Executing tool: {fc.name}")
                        yield f"data: {json.dumps({'status': f'Using tool: {fc.name}'})}\n\n"
                        
                        args = fc.args
                        # Inject user_id for Google Drive tools
                        if fc.name in ["list_google_drive_folders", "create_google_drive_folder"]:
                            print(f"Injecting user_id for {fc.name}")
                            args["user_id"] = user_id
                        
                        try:
                            # Execute tool using MCP manager
                            result = await mcp_manager.call_tool_by_name(fc.name, args)
                            
                            # Format result
                            response_content = result if isinstance(result, str) else str(result)
                            if isinstance(result, list) and len(result) > 0 and hasattr(result[0], 'text'):
                                response_content = result[0].text
                                
                            response_parts.append(types.Part.from_function_response(
                                name=fc.name,
                                response={"result": response_content}
                            ))
                        except Exception as e:
                            print(f"Tool execution failed: {e}")
                            response_parts.append(types.Part.from_function_response(
                                name=fc.name,
                                response={"error": str(e)}
                            ))
                            
                    contents.append(types.Content(role="user", parts=response_parts))
                
                yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id})}\n\n"

            # Save assistant message
            await messages_collection.insert_one({
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now()
            })
                
        except Exception as e:
            print(f"Multimodal chat stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
