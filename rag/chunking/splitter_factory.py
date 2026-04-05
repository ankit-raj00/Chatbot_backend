from typing import Dict, Any
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Note: SemanticChunker is experimental, checking import availability or falling back
try:
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:
    SemanticChunker = None

import os

class SplitterFactory:
    """
    Implements the 'Chunking Strategy' column of the Ingestion Matrix.
    """
    
    @staticmethod
    def get_splitter(strategy: str, config: Dict[str, Any]):
        """
        Factory method to return the appropriate LangChain splitter.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔪 Creating Splitter: [Strategy: {strategy}] [Config: {config}]")
        
        # --- 1. Recursive (Baseline / Textbooks / Large Window) ---
        if strategy in ["recursive", "recursive_large"]:
            return RecursiveCharacterTextSplitter(
                chunk_size=config.get("chunk_size", 512),
                chunk_overlap=config.get("overlap", 50),
                separators=["\n\n", "\n", " ", ""]
            )
            
        # --- 2. Hierarchical (Legal) ---
        if strategy == "hierarchical":
            # Using Recursive but with markdown headers as priority
            return RecursiveCharacterTextSplitter(
                chunk_size=config.get("chunk_size", 1024),
                chunk_overlap=config.get("overlap", 100),
                separators=config.get("separators", ["\n## ", "\n### ", "\n", " "])
            )

        # --- 3. Code Splitter (Tech Manuals) ---
        if strategy == "code_splitter":
            lang_map = {
                "py": Language.PYTHON,
                "js": Language.JS,
                "ts": Language.JS,
                "java": Language.JAVA,
                "cpp": Language.CPP,
            }
            lang_str = config.get("language", "py")
            language = lang_map.get(lang_str, Language.PYTHON)
            
            return RecursiveCharacterTextSplitter.from_language(
                language=language, 
                chunk_size=512, 
                chunk_overlap=50
            )

        # --- 4. Semantic (Academic / VLM) ---
        if strategy == "semantic":
            if not SemanticChunker:
                # Fallback if experimental not installed
                return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Guardrail C: Hard Chunk Caps
            # Note: SemanticChunker logic is internal, but we can wrap it or expect 
            # the router config "max_chunk_size" to be enforced post-splitting if needed.
            return SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile"
            )

        # --- 5. Document Based (Resume - Whole Doc) ---
        if strategy == "document_based":
             # If chunk_size is 0, we want the whole document.
             # RecursiveCharacterTextSplitter with a huge limit works.
             size = config.get("chunk_size", 4000)
             if size == 0:
                 size = 10000 # Effectively infinite for a resume
                 
             return RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=0,
                separators=["\n\n", "\n"] 
             )

        # --- Default ---
        return RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
