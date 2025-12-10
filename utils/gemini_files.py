import os
from google import genai
from google.genai import types
import asyncio
from functools import partial

class GeminiFiles:
    """Helper for Gemini Files API"""
    
    def __init__(self, client: genai.Client):
        self.client = client

    async def upload_file(self, file_path: str, mime_type: str = None):
        """
        Upload file to Gemini Files API (Async wrapper)
        """
        # Run synchronous upload in thread pool
        loop = asyncio.get_running_loop()
        func = partial(self.client.files.upload, file=file_path, config={'mime_type': mime_type} if mime_type else None)
        uploaded_file = await loop.run_in_executor(None, func)
        return uploaded_file

    async def delete_file(self, file_name: str):
        """
        Delete file from Gemini Files API (Async wrapper)
        """
        loop = asyncio.get_running_loop()
        func = partial(self.client.files.delete, name=file_name)
        await loop.run_in_executor(None, func)
