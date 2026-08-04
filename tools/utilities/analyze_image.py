"""
analyze_image — the agent's vision tool.

Modern "vision sub-call" architecture: the main agent calls
analyze_image(sandbox_path, query); this tool loads the image from the sandbox,
makes an ISOLATED vision LLM call with the image as a base64 data URI in a USER
message, and returns the model's text answer. This is required because the LLM
gateway (OmniRoute -> Gemini) only accepts image input in a user message — it
ignores images placed in tool-role messages and rejects plain image URLs. So we
can't hand the raw image back as a tool result; instead we do the "looking"
inside the tool and return what was seen.

Works for ANY image in the sandbox — user uploads (uploads/…), files the agent
generated (outputs/…, work/…) — on any turn. This lets the agent visually verify
its own output: render a chart/PDF page to PNG, then analyze_image it to check
the result before delivering.
"""
import asyncio
import base64
import io
import mimetypes
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from utils.workspace import conversation_workspace_for, is_path_within_conversation_sandbox
from config.model_config import ModelConfig

# Long-edge cap — images larger than this are downscaled before the vision call
# to keep the request small/fast (this is well within typical vision limits and
# is plenty of detail for description/OCR).
_MAX_DIM = 1536
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")


def _prepare_data_uri(raw: bytes, suffix: str) -> str:
    """Return a base64 data URI, downscaling oversized images. Falls back to the
    raw bytes if PIL can't process them."""
    mime = mimetypes.types_map.get(suffix.lower(), "image/png")
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if max(w, h) > _MAX_DIM:
            scale = _MAX_DIM / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            if img.mode not in ("RGB", "RGBA", "L"):
                img = img.convert("RGB")
            out = io.BytesIO()
            img.save(out, format="PNG")
            raw = out.getvalue()
            mime = "image/png"
    except Exception:
        pass  # not decodable by PIL — send original bytes as-is
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def make_analyze_image_tool(user_id: str, conversation_id: str):
    ws_root = conversation_workspace_for(user_id, conversation_id)

    _DEFAULT_QUERY = ("Describe this image in thorough detail: any text (quoted verbatim), "
                      "objects, people, layout, colors, and notable details.")

    class AnalyzeImageInput(BaseModel):
        sandbox_path: str = Field(description="Sandbox-relative path to the image, e.g. "
                                              "'uploads/photo.jpg' or 'outputs/chart.png'.")
        query: str = Field(default=_DEFAULT_QUERY,
                           description="What you want to know about the image. Be specific for "
                                       "targeted questions (e.g. 'What is the hex color of the header?', "
                                       "'Transcribe all text', 'Is the chart's title centered?').")

    async def analyze_image(sandbox_path: str, query: str = _DEFAULT_QUERY) -> str:
        """
        Look at an image with vision and answer a question about it.

        Use this for ANY image you need to SEE — a picture the user uploaded, or
        one you generated yourself (e.g. render a chart or a PDF page to PNG, then
        analyze_image it to visually verify the result before finishing).

        Returns a text description/answer, not the raw image.
        """
        if not is_path_within_conversation_sandbox(user_id, conversation_id, sandbox_path):
            return "BLOCKED: path outside sandbox"

        target = (ws_root / sandbox_path).resolve() if not Path(sandbox_path).is_absolute() else Path(sandbox_path).resolve()
        if not target.exists() or not target.is_file():
            return f"Error: image not found at '{sandbox_path}'."
        if target.suffix.lower() not in _IMAGE_EXTS:
            return (f"Error: '{sandbox_path}' does not look like an image "
                    f"({target.suffix or 'no extension'}). Read non-image files with run_python/run_shell.")

        try:
            raw = await asyncio.to_thread(target.read_bytes)
        except Exception as e:
            return f"Error reading image '{sandbox_path}': {e}"

        data_uri = await asyncio.to_thread(_prepare_data_uri, raw, target.suffix)

        from graph.llm_registry import get_llm
        llm = get_llm(ModelConfig.VISION_MODEL)
        try:
            resp = await llm.ainvoke([HumanMessage(content=[
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ])])
        except Exception as e:
            return f"Vision call failed for '{sandbox_path}': {e}"

        content = resp.content
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return str(content) or "(the vision model returned no text)"

    return StructuredTool.from_function(
        coroutine=analyze_image,
        name="analyze_image",
        description="Look at an image (uploaded OR one you generated) with vision and answer a "
                    "question about it. Returns a text answer. Use it to inspect user images and "
                    "to visually verify your own rendered output (charts, PDF pages, etc.).",
        args_schema=AnalyzeImageInput,
    )
