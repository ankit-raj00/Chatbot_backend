import os
import re

file_path = "utils/mcp_connection_manager.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Add import if missing
if "import structlog" not in content:
    content = re.sub(r'(import time\n)', r'\1import structlog\nlogger = structlog.get_logger(__name__)\n', content, count=1)

# Replace simple prints with logger.info
content = re.sub(r'print\(f"(.*?)"\)', r'logger.info(f"\1")', content)
content = re.sub(r'print\("(.*?)"\)', r'logger.info("\1")', content)
content = re.sub(r'print\(f\'(.*?)\'\)', r'logger.info(f"\1")', content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated utils/mcp_connection_manager.py")
