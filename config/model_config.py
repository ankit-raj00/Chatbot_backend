"""
Model configuration for Gemini models
Centralized place for all model-related settings
"""

from typing import Dict, List, Any

class ModelConfig:
    """Configuration for available Gemini models"""
    
    # Available models with their capabilities
    MODELS: Dict[str, Dict[str, Any]] = {
        "gemini-2.5-flash-lite": {
            "name": "Gemini 2.5 Flash Lite",
            "description": "Fastest, highest quota — primary model for all features",
            "supports_images": True,
            "supports_video": True,
            "supports_audio": True,
            "max_tokens": 8192,
            "context_window": 1000000,
        },
        "gemini-2.5-pro": {
            "name": "Gemini 2.5 Pro",
            "description": "Most capable model, best for complex tasks",
            "supports_images": True,
            "supports_video": True,
            "supports_audio": True,
            "max_tokens": 8192,
            "context_window": 1000000,
        },
        "gemini-2.5-flash": {
            "name": "Gemini 2.5 Flash",
            "description": "Fast and efficient, good for most tasks",
            "supports_images": True,
            "supports_video": True,
            "supports_audio": True,
            "max_tokens": 8192,
            "context_window": 1000000,
        },

        "gemini-3.1-flash-lite": {
            "name": "Gemini 3.1 Flash Lite",
            "description": "Latest fast model",
            "supports_images": True,
            "supports_video": True,
            "supports_audio": True,
            "max_tokens": 8192,
            "context_window": 1000000,
        },
        "gemini-flash-latest": {
            "name": "Gemini Flash (Latest)",
            "description": "Latest flash model with newest features",
            "supports_images": True,
            "supports_video": True,
            "supports_audio": True,
            "max_tokens": 8192,
            "context_window": 1000000,
        },
    }

    # Default model — highest quota, lowest latency
    DEFAULT_MODEL = "gemini-3.1-flash-lite"
    
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
