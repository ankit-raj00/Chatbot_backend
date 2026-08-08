"""
Model configuration.

Models are served through OmniRoute (an OpenAI-compatible gateway), so IDs use
OmniRoute's `provider/model` naming (e.g. "antigravity/gemini-3.5-flash-medium").
Add more entries here to expose them in the frontend model selector.

Per-model USD pricing is optional; when unknown (default 0.0) the app reports a
0 cost rather than a wrong one.
"""

from typing import Dict, List, Any

class ModelConfig:
    """Configuration for available models (served via OmniRoute)."""

    # Available models with their capabilities.
    # `price_in`/`price_out` are USD per 1M tokens (0 => unknown/not billed).
    MODELS: Dict[str, Dict[str, Any]] = {
        "antigravity/gemini-3.5-flash-medium": {
            "name": "Gemini 3.5 Flash (Medium)",
            "description": "Default model via OmniRoute — fast, multimodal, tool-capable",
            "supports_images": True,
            "supports_video": True,
            "supports_audio": True,
            "max_tokens": 8192,
            "context_window": 1000000,
            # Google's official standard-tier Gemini 3.5 Flash pricing, USD per
            # 1M tokens (output includes thinking tokens). OmniRoute reports
            # $0 for every call through the free "antigravity" provider pool,
            # so this is what turns real token counts into a real cost.
            "price_in": 1.50,
            "price_out": 9.00,
        },
    }

    # Default model — served via OmniRoute
    DEFAULT_MODEL = "antigravity/gemini-3.5-flash-medium"

    # Model used by MemoryService to extract durable facts silently in the background
    MEMORY_EXTRACTION_MODEL = "antigravity/gemini-3.5-flash-medium"

    # Model used by the analyze_image vision tool (must support image input)
    VISION_MODEL = "antigravity/gemini-3.5-flash-medium"

    @classmethod
    def get_pricing(cls, model_id: str) -> Dict[str, float]:
        """Return {'input': usd_per_token, 'output': usd_per_token} for a model.
        Defaults to 0.0 when the model or its pricing is unknown."""
        info = cls.MODELS.get(model_id, {})
        return {
            "input":  float(info.get("price_in", 0.0)) / 1_000_000,
            "output": float(info.get("price_out", 0.0)) / 1_000_000,
        }

    @classmethod
    def get_model_info(cls, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        return cls.MODELS.get(model_id, cls.MODELS[cls.DEFAULT_MODEL])
    
    @classmethod
    def get_all_models(cls) -> List[Dict[str, Any]]:
        """Get list of all available models"""
        return [
            {"id": model_id, **info}
            for model_id, info in cls.MODELS.items()
        ]
    
    @classmethod
    def is_valid_model(cls, model_id: str) -> bool:
        """Check if model ID is valid"""
        return model_id in cls.MODELS
    
    @classmethod
    def supports_images(cls, model_id: str) -> bool:
        """Check if model supports image input"""
        model_info = cls.get_model_info(model_id)
        return model_info.get("supports_images", False)
