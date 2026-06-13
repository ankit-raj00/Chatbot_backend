import os
import re

file_path = "e2e_test.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

content = re.sub(
    r'BASE_URL\s*=\s*"https://chatbot-backend-jsfm\.onrender\.com"',
    'from dotenv import load_dotenv\nload_dotenv()\nBASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")',
    content
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated e2e_test.py")
