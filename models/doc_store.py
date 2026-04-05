from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import List, Optional, Dict, Any

class ImageSummary(BaseModel):
    image_file: str
    image_path: str
    cloudinary_url: Optional[str] = None
    summary: str

class PageContent(BaseModel):
    page: int
    text: str
    markdown: str
    images: List[ImageSummary] = []

class ParsedDocument(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    
    id: Optional[str] = Field(None, alias="_id")
    document_id: str
    source: Dict[str, Any]
    pages: List[PageContent]
    created_at: datetime = Field(default_factory=datetime.now)
