"""
File handling utilities for multimodal support
Handles image upload, validation, and storage
"""

import base64
import os
import mimetypes
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException, status
from PIL import Image
import io

class FileHandler:
    """Utilities for handling file uploads"""
    
    # Allowed image types
    ALLOWED_IMAGE_TYPES = {
        "image/jpeg",
        "image/jpg", 
        "image/png",
        "image/gif",
        "image/webp",
        "application/pdf"
    }
    
    # Max file size (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    @classmethod
    async def validate_image(cls, file: UploadFile) -> None:
        """Validate uploaded image file"""
        # Check content type
        if file.content_type not in cls.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed types: {', '.join(cls.ALLOWED_IMAGE_TYPES)}"
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > cls.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Max size: {cls.MAX_FILE_SIZE / (1024*1024)}MB"
            )
    
    @classmethod
    async def encode_image_to_base64(cls, file: UploadFile) -> Tuple[str, str]:
        """
        Encode image to base64 string
        Returns: (base64_string, mime_type)
        """
        await cls.validate_image(file)
        
        # Read file content
        content = await file.read()
        
        # Encode to base64
        base64_string = base64.b64encode(content).decode('utf-8')
        
        return base64_string, file.content_type
    
    @classmethod
    def decode_base64_image(cls, base64_string: str) -> bytes:
        """Decode base64 string to image bytes"""
        return base64.b64decode(base64_string)
    
    @classmethod
    async def get_image_dimensions(cls, file: UploadFile) -> Tuple[int, int]:
        """Get image width and height"""
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        image = Image.open(io.BytesIO(content))
        return image.size  # (width, height)
    
    @classmethod
    def get_mime_type(cls, filename: str) -> Optional[str]:
        """Get MIME type from filename"""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type
