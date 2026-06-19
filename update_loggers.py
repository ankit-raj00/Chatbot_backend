import os
import re

directories_to_scan = [
    "controllers",
    "services",
    "rag",
    "utils",
    "graph"
]

files_to_update = [
    "controllers/chat_controller.py",
    "services/chat_service.py",
    "services/history_service.py",
    "services/memory_service.py",
    "services/ingestion_job_service.py",
    "rag/ingestion_service.py",
    "rag/parsers/llama_parse_client.py",
    "rag/vector_store/qdrant_manager.py",
    "rag/graph/nodes/grader_node.py",
    "rag/graph/nodes/retrieval_node.py",
    "rag/graph/nodes/hallucination_node.py",
    "rag/graph/workflow.py",
    "rag/graph/nodes/agent_node.py",
    "rag/tools/retrieval_tool.py",
    "utils/mcp_connection_manager.py",
    "utils/hooks.py"
]

for file_path in files_to_update:
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}, does not exist.")
        continue
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace 'import logging' with 'import structlog' (if it's just 'import logging' and not 'from logging import ...')
    # Actually, it's safer to just add 'import structlog' and replace the getLogger line.
    
    new_content = re.sub(
        r'logger\s*=\s*logging\.getLogger\(__name__\)',
        r'import structlog\nlogger = structlog.get_logger(__name__)',
        content
    )
    
    # In some files it might be logger = logging.getLogger("...")
    new_content = re.sub(
        r'logger\s*=\s*logging\.getLogger\((["\'].*?["\'])\)',
        r'import structlog\nlogger = structlog.get_logger(\1)',
        new_content
    )
    
    if content != new_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {file_path}")
    else:
        print(f"No changes made to {file_path}")
