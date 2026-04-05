from typing import Dict, Any, Optional, List
import mimetypes
import os

class IngestionRouter:
    """
    The 'Smart Router' that directs files to the optimal pipeline based on Cost vs Accuracy.
    Implements the 13-Type Matrix from Architecture v11.1.
    """
    
    def __init__(self):
        # Configuration for "Production Guardrails"
        self.MAX_SEMANTIC_CHUNK_SIZE = 2000
        self.ENABLE_VLM_THRESHOLD = 0.10 # 10% image density
        
    def get_supported_categories(self) -> List[str]:
        """Returns list of supported doc types for UI dropdown."""
        return [
            "Auto (Detect)", "Financial", "Legal", "Academic", "Textbook", "Resume", 
            "Medical", "Slides", "Code Manual", "Email", "CAD", "Chat Log", "Fiction", "General"
        ]

    def route(self, file_name: str, file_type: str, file_size: int, force_category: Optional[str] = None) -> Dict[str, Any]:
        """
        Determines the parsing and chunking strategy.
        If force_category is provided, overrides heuristics.
        """
        # --- 0. Manual Override 🛠️ ---
        if force_category and force_category != "Auto (Detect)":
             config = self._get_config_for_category(force_category, file_name)
             if config:
                 config["rationale"] += " (User Selected)"
                 return config

        extension = os.path.splitext(file_name)[1].lower()
        
        # --- 1. Finance (PDF) 📊 ---
        if self._is_financial(file_name):
            return self._get_config_for_category("Financial", file_name)

        # --- 2. Legal (PDF) ⚖️ ---
        if self._is_legal(file_name):
             return self._get_config_for_category("Legal", file_name)

        # --- 3. Academic (PDF) 🔬 ---
        if self._is_academic(file_name):
             return self._get_config_for_category("Academic", file_name)
            
        # --- 5. Textbook (PDF) 📚 ---
        if "textbook" in file_name.lower():
             return self._get_config_for_category("Textbook", file_name)

        # --- 6. Resume (PDF) 👤 ---
        if "resume" in file_name.lower() or "cv" in file_name.lower():
             return self._get_config_for_category("Resume", file_name)
            
         # --- 8. Medical (PDF) 🏥 ---
        if self._is_medical(file_name):
             return self._get_config_for_category("Medical", file_name)

        # --- 7. Slide Deck (PPTX) 🖥️ ---
        if extension == ".pptx":
            return self._get_config_for_category("Slides", file_name)
            
        # --- 9. Tech Manual (MD/Code) ⚙️ ---
        if extension in [".md", ".py", ".js", ".ts", ".java", ".cpp", ".json"]:
             return self._get_config_for_category("Code Manual", file_name)
        
        # --- 10. Email (PST/MBOX) 📧 ---
        if extension in [".pst", ".mbox", ".eml"]:
             return self._get_config_for_category("Email", file_name)

        # --- 11. CAD (DXF) 📐 ---
        if extension == ".dxf":
             return self._get_config_for_category("CAD", file_name)
            
        # --- 12. Chat (JSON) 💬 ---
        if "chat" in file_name.lower() and extension == ".json":
             return self._get_config_for_category("Chat Log", file_name)

        # --- 4. Novel (Fiction) 🧙‍♂️ ---
        if self._is_fiction(file_name) or extension == ".epub":
             return self._get_config_for_category("Fiction", file_name)

        # --- 13. General / Baseline 📄 ---
        return self._get_config_for_category("General", file_name)

    def _get_config_for_category(self, category: str, file_name: str) -> Optional[Dict[str, Any]]:
        """Helper to return config dict based on category name."""
        category = category.lower()
        extension = os.path.splitext(file_name)[1].lower()

        if category == "financial":
             return {
                "type_category": "financial",
                "parser_strategy": "llama_parse",
                "chunking_strategy": "parent_document",
                "parser_config": {"mode": "markdown", "continuous_mode": True, "premium_mode": True},
                "chunker_config": {"parent_chunk_size": 4000, "child_chunk_size": 200, "mode": "parent_document"},
                "rationale": "Atomic Tables required."
            }
        elif category == "legal":
            return {
                "type_category": "legal",
                "parser_strategy": "llama_parse",
                "chunking_strategy": "hierarchical",
                 "parser_config": {"mode": "markdown", "instruction": "Preserve all section headers and numbering.", "split_by_markdown_header": True},
                 "chunker_config": {"chunk_size": 1024, "overlap": 100, "separators": ["\n## ", "\n### ", "\n#### "]},
                "rationale": "Clause integrity."
            }
        elif category == "academic":
             return {
                "type_category": "academic",
                "parser_strategy": "llama_parse",
                "chunking_strategy": "semantic",
                "parser_config": {"mode": "markdown", "use_vendor_multimodal_model": True, "instruction": "Describe all charts and formulas."},
                "chunker_config": {"breakpoint_threshold_type": "percentile", "max_chunk_size": 2000},
                "rationale": "Semantic chunking."
            }
        elif category == "textbook":
             return {
                "type_category": "textbook",
                "parser_strategy": "llama_parse",
                "chunking_strategy": "hybrid_chapter",
                "parser_config": {"mode": "markdown"},
                "chunker_config": {"split_on": "chapter"},
                "rationale": "Topic-based search."
            }
        elif category == "resume":
              return {
                "type_category": "resume",
                "parser_strategy": "llama_parse",
                "chunking_strategy": "document_based",
                "parser_config": {"mode": "markdown", "instruction": "Extract skills and experience as a structured profile."},
                "chunker_config": {"chunk_size": 0}, 
                "rationale": "Holistic profile."
            }
        elif category == "medical":
            return {
                "type_category": "medical",
                "parser_strategy": "llama_parse",
                "chunking_strategy": "parent_document",
                "parser_config": {"mode": "markdown", "is_confidential": True},
                "chunker_config": {"parent_chunk_size": 2000, "child_chunk_size": 256},
                 "rationale": "Privacy & Massive Tables."
            }
        elif category == "slides":
             return {
                "type_category": "slides",
                "parser_strategy": "unstructured",
                "chunking_strategy": "page_based",
                "parser_config": {"strategy": "hi_res"},
                "chunker_config": {"split_strategy": "by_page"},
                "rationale": "1 Slide = 1 Chunk."
            }
        elif category == "code manual":
              return {
                "type_category": "code_manual",
                "parser_strategy": "unstructured",
                "chunking_strategy": "code_splitter",
                "parser_config": {},
                "chunker_config": {"language": extension.replace(".", "") if extension else "py"},
                "rationale": "Preserves AST/Indentation."
            }
        elif category == "email":
             return {
                "type_category": "email",
                "parser_strategy": "libratom",
                "chunking_strategy": "thread_based",
                "parser_config": {},
                "chunker_config": {"group_by": "thread_id"},
                "rationale": "Keep Reply-Chain together."
            }
        elif category == "cad":
            return {
                "type_category": "cad",
                "parser_strategy": "ezdxf",
                "chunking_strategy": "recursive",
                "chunker_config": {"chunk_size": 1000},
                "rationale": "Convert Geometry to Text."
            }
        elif category == "chat log":
              return {
                "type_category": "chat_log",
                "parser_strategy": "json",
                "chunking_strategy": "time_window",
                "parser_config": {"time_field": "timestamp"},
                "chunker_config": {"window_minutes": 5},
                "rationale": "Group 5-minute conversation bursts."
            }
        elif category == "fiction":
              return {
                "type_category": "fiction",
                "parser_strategy": "unstructured",
                "chunking_strategy": "recursive",
                "parser_config": {"strategy": "fast"},
                "chunker_config": {"chunk_size": 1000, "overlap": 200},
                "rationale": "Narrative flow."
            }
        else: # General
             return {
                "type_category": "general",
                "parser_strategy": "unstructured",
                "chunking_strategy": "recursive",
                "parser_config": {"strategy": "fast"},
                "chunker_config": {"chunk_size": 512, "overlap": 50},
                "rationale": "Baseline."
             }

    # --- Heuristics Helpers ---
    
    def _is_financial(self, name: str) -> bool:
        keywords = ["balance", "p&l", "financial", "statement", "invoice", "receipt", "10k", "10q", "quarterly"]
        return any(k in name.lower() for k in keywords)

    def _is_legal(self, name: str) -> bool:
        keywords = ["contract", "agreement", "nda", "terms", "policy", "regulation", "compliance", "law"]
        return any(k in name.lower() for k in keywords)

    def _is_academic(self, name: str) -> bool:
        keywords = ["paper", "journal", "thesis", "dissertation", "study", "research", "arxiv"]
        return any(k in name.lower() for k in keywords)

    def _is_medical(self, name: str) -> bool:
        keywords = ["lab", "medical", "patient", "clinical", "report", "blood", "scan"]
        return any(k in name.lower() for k in keywords)
        
    def _is_fiction(self, name: str) -> bool:
        keywords = ["novel", "story", "book", "fiction", "chapter"]
        return any(k in name.lower() for k in keywords)
