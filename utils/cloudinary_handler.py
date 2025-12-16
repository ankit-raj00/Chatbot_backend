import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
import requests
import tempfile
import asyncio
from typing import Tuple, Optional

class CloudinaryHandler:
    """Handle file uploads and downloads with Cloudinary"""
    
    def __init__(self):
        """Initialize Cloudinary configuration from environment variables"""
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True
        )
    
    async def upload_file(self, file_path: str, folder: str = "chatbot/files") -> Tuple[str, str]:
        """
        Upload a file to Cloudinary asynchronously
        
        Args:
            file_path: Path to the file to upload
            folder: Cloudinary folder path (default: chatbot/files)
        
        Returns:
            Tuple of (cloudinary_url, public_id)
        """
        try:
            # Upload file with resource_type auto-detection in a separate thread
            result = await asyncio.to_thread(
                cloudinary.uploader.upload,
                file_path,
                folder=folder,
                resource_type="auto",  # Auto-detect file type
                use_filename=True,
                unique_filename=True
            )
            
            return result["secure_url"], result["public_id"]
        except Exception as e:
            print(f"Cloudinary upload error: {e}")
            raise
    
    async def download_file(self, cloudinary_url: str) -> str:
        """
        Download a file from Cloudinary to a temporary location asynchronously
        
        Args:
            cloudinary_url: The Cloudinary URL to download from
        
        Returns:
            Path to the downloaded temporary file
        """
        try:
            # Download file in a separate thread
            def _download():
                response = requests.get(cloudinary_url, stream=True)
                response.raise_for_status()
                
                # Extract file extension from URL
                extension = cloudinary_url.split('.')[-1].split('?')[0]
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    return tmp.name

            return await asyncio.to_thread(_download)
        except Exception as e:
            print(f"Cloudinary download error: {e}")
            raise
    
    async def delete_file(self, public_id: str) -> bool:
        """
        Delete a file from Cloudinary asynchronously
        
        Args:
            public_id: The Cloudinary public_id of the file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete file in a separate thread
            result = await asyncio.to_thread(
                cloudinary.uploader.destroy,
                public_id,
                resource_type="auto"
            )
            return result.get("result") == "ok"
        except Exception as e:
            print(f"Cloudinary delete error: {e}")
            return False
