import os
import shutil
import logging
import json
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import nest_asyncio
from dotenv import load_dotenv

# Parsing Libraries
from llama_parse import LlamaParse
from langchain_core.documents import Document

# Vision & Cloud Libraries
from google import genai
from PIL import Image
from utils.cloudinary_handler import CloudinaryHandler
from core.database import doc_store_collection
from models.doc_store import ParsedDocument, PageContent, ImageSummary

# Apply nest_asyncio
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback Import
try:
    from langchain_unstructured import UnstructuredLoader
except ImportError:
    try:
        from langchain_community.document_loaders import UnstructuredLoader
    except ImportError:
        try:
            from langchain_community.document_loaders import UnstructuredFileLoader as UnstructuredLoader
        except ImportError:
            UnstructuredLoader = None

def obj_to_dict(obj):
    if obj is None: return None
    if isinstance(obj, dict): return {k: obj_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list): return [obj_to_dict(x) for x in obj]
    if hasattr(obj, "to_dict"): return obj.to_dict()   # LlamaIndex common
    if hasattr(obj, "model_dump"): return obj.model_dump() # Pydantic v2
    if hasattr(obj, "dict"): return obj.dict()        # Pydantic v1
    if hasattr(obj, "__dict__"): return obj.__dict__  # Standard class
    return obj

class LlamaParseClient:
    """
    Advanced Multi-Modal Parser.
    Matches User's Script Logic: LlamaParse (Agentic) -> Gemini Vision -> Cloudinary -> Markdown Compiler.
    """
    
    def __init__(self):
        self.api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if self.google_api_key:
            self.genai_client = genai.Client(api_key=self.google_api_key)
        else:
            self.genai_client = None

        self.cloudinary = CloudinaryHandler()

        if not self.api_key:
            logger.warning("LLAMA_CLOUD_API_KEY not found. Fallback forced.")

    async def parse(self, file_path: str, config: Dict[str, Any]) -> List[Document]:
        if not self.api_key:
            return self._fallback_unstructured(file_path)

        try:
            logger.info(f"🚀 Starting Advanced Parsing for: {os.path.basename(file_path)}")
            
            # 1. Initialize Parser (Exact params from user script)
            parser = LlamaParse(
                api_key=self.api_key,
                page_separator="\n\n---\n\n",
                output_tables_as_HTML=True,
                precise_bounding_box=False,
                tier="agentic",
                version="latest",
                verbose=True,
            )

            # 2. Parse (aparse return JobResult)
            llama_result = await parser.aparse(file_path)

            # 3. Extract Images
            image_dir = f"temp_images_{os.path.basename(file_path)}"
            os.makedirs(image_dir, exist_ok=True)
            
            logger.info("📸 Extracting images...")
            images = await llama_result.aget_image_documents(
                include_screenshot_images=False,
                include_object_images=True,
                image_download_dir=image_dir
            )
            
            # 4. Process Images (Summarize + Upload)
            # Strategy: Iterate through the downloaded files (User Script Pattern)
            # This avoids 'ImageDocument' attribute errors
            
            image_map = {} # {filename: {url, summary}}
            available_images = set()

            logger.info("🔍 Processing extracted images from disk...")
            
            if os.path.exists(image_dir):
                for fname in os.listdir(image_dir):
                    # Filter for image extensions
                    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    local_path = os.path.join(image_dir, fname)
                    available_images.add(fname)
                    
                    # A. Summarize (Gemini)
                    # Run in parallel if needed, but sequential is safer for rate limits
                    summary = await self._summarize_image(local_path)
                    
                    # B. Upload (Cloudinary)
                    try:
                        url, _ = await self.cloudinary.upload_file(local_path)
                    except Exception as e:
                        logger.error(f"Cloudinary upload error for {fname}: {e}")
                        url = None

                    image_map[fname] = {
                        "summary": summary,
                        "url": url,
                        "local_path": local_path
                    }
            else:
                 logger.warning(f"Image directory {image_dir} was not created.")

            # 5. Convert Pages to JSON
            pages_json = []
            for page in llama_result.pages:
                pages_json.append({
                    "page": page.page,
                    "text": page.text,
                    "md": page.md,
                    "images": [obj_to_dict(img) for img in getattr(page, "images", [])],
                    "items": [obj_to_dict(item) for item in getattr(page, "items", [])]
                })

            # 6. Build Markdown (The Compiler)
            final_markdown = self._build_markdown_from_json(pages_json, image_map, available_images)
            
            # 7. Save Structured JSON (DocStore) -> MongoDB
            doc_store_id = await self._save_json_doc_store(pages_json, image_map, available_images, file_path)
            
            # --- DEBUG: Save Final Markdown to Disk ---
            debug_dir = os.path.join(os.path.dirname(__file__), "debug_parsed_md")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"final_{os.path.basename(file_path)}.md")
            
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(final_markdown)
            logger.info(f"✅ Final Markdown saved: {debug_path}")
            # ------------------------------------------
            
            # 8. Cleanup Temporary Files
            if os.path.exists(image_dir):
                shutil.rmtree(image_dir)
                logger.info(f"🧹 Temporary images cleaned up: {image_dir}")

            # 9. Return Single Document
            return [Document(
                page_content=final_markdown,
                metadata={
                    "source": file_path,
                    "parser": "llama_parse_v2_vision",
                    "images_processed": len(available_images),
                    "json_id": doc_store_id # Link to MongoDB DocStore
                }
            )]

        except Exception as e:
            logger.error(f"Advanced Parsing Failed: {e}")
            logger.info("Triggering Fallback: Unstructured (Local)")
            return self._fallback_unstructured(file_path)

    async def _summarize_image(self, image_path: str) -> str:
        if not self.genai_client: return "Image content"
        try:
            return await asyncio.to_thread(self._run_inference, image_path)
        except Exception:
            return "Image analysis failed"

    def _run_inference(self, image_path: str) -> str:
        # EXACT PROMPT FROM USER SCRIPT
        img = Image.open(image_path)
        prompt = (
            "Describe this image in 1–2 concise sentences. "
            "Focus only on what is visibly present. "
            "Do not speculate or infer hidden meaning."
        )
        response = self.genai_client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[prompt, img]
        )
        return response.text.strip()

    def _render_item(self, item):
        t = item.get("type")
        # Ensure we return strings, not None
        if t == "heading": return str(item.get("md", ""))
        if t == "text": return str(item.get("value", ""))
        if t == "table": return str(item.get("md", ""))
        return ""

    def _build_markdown_from_json(self, pages, image_map, available_images):
        md_out = []

        for page in pages:
            md_out.append(f"\n<!-- Page {page.get('page')} -->\n")

            items = page.get("items", [])
            images = sorted(page.get("images", []), key=lambda x: x.get("y", 0))
            img_idx = 0

            for item in items:
                md_out.append(self._render_item(item))

                while img_idx < len(images):
                    img = images[img_idx]
                    fname = img.get("name")
                    
                    if fname not in available_images:
                        img_idx += 1
                        continue

                    # Logic: if img["y"] > item["bBox"]["y"]
                    item_y = 0
                    if "bBox" in item and item["bBox"]:
                        item_y = item["bBox"].get("y", 0)
                    
                    img_y = img.get("y", 0)

                    if img_y > item_y:
                        info = image_map.get(fname, {})
                        summary = info.get("summary", "Image content")
                        url = info.get("url", "")
                        
                        if url:
                            md_out.append(
                                f"\n\n**Image:** `{fname}`\n\n"
                                f"**Summary:** {summary}\n\n"
                                f"![]({url})\n"
                            )
                        img_idx += 1
                    else:
                        break

            # remaining images
            for img in images[img_idx:]:
                fname = img.get("name")
                if fname not in available_images: continue
                
                info = image_map.get(fname, {})
                summary = info.get("summary", "Image content")
                url = info.get("url", "")
                
                if url:
                    md_out.append(
                        f"\n\n**Image:** `{fname}`\n\n"
                        f"**Summary:** {summary}\n\n"
                        f"![]({url})\n"
                    )

        return "\n\n".join(md_out)

    async def _save_json_doc_store(self, pages_json, image_map, available_images, file_path) -> str:
        """
        Save parsed document to MongoDB (DocStore) and return the ObjectId as string.
        """
        try:
            filename = os.path.basename(file_path)
            doc_id = os.path.splitext(filename)[0]
            
            # Construct Pages using Pydantic Models for validation
            page_contents = []
            for page in pages_json:
                images_list = []
                for img in page.get("images", []):
                    img_name = img.get("name")
                    if img_name in available_images:
                        info = image_map.get(img_name, {})
                        images_list.append(ImageSummary(
                            image_file=img_name,
                            image_path=info.get("local_path", ""),
                            cloudinary_url=info.get("url"),
                            summary=info.get("summary", "Image content")
                        ))
                
                page_contents.append(PageContent(
                    page=page.get("page"),
                    text=page.get("text", ""),
                    markdown=page.get("md", ""),
                    images=images_list
                ))

            # Create Document Model
            doc = ParsedDocument(
                document_id=doc_id,
                source={
                    "filename": filename,
                    "parser": "llama_parse_v2",
                    "created_at": datetime.now().isoformat()
                },
                pages=page_contents
            )
            
            # Save to MongoDB
            result = await doc_store_collection.insert_one(doc.model_dump(by_alias=True, exclude={"id"}))
            mongo_id = str(result.inserted_id)
            logger.info(f"💾 DocStore saved to MongoDB: {mongo_id}")
            
            return mongo_id

        except Exception as e:
            logger.error(f"Failed to save DocStore to MongoDB: {e}")
            return "error_saving_json"

    def _fallback_unstructured(self, file_path: str) -> List[Document]:
        if not UnstructuredLoader: raise ImportError("UnstructuredLoader unavailable.")
        try:
            loader = UnstructuredLoader(file_path, strategy="fast", mode="single")
            docs = loader.load()
            for doc in docs:
                doc.metadata["parser"] = "unstructured_fallback"
            return docs
        except Exception as e:
            logger.error(f"Fallback failed: {e}")
            raise e
