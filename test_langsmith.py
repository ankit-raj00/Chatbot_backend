import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langsmith import Client

load_dotenv()

print("Testing LangSmith Connection...")
print(f"LANGCHAIN_TRACING_V2: {os.environ.get('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_PROJECT: {os.environ.get('LANGCHAIN_PROJECT')}")
api_key = os.environ.get('LANGCHAIN_API_KEY', '')
print(f"LANGCHAIN_API_KEY present: {bool(api_key)} (Starts with: {api_key[:8]}...)")

try:
    # First test: Try to directly ping the LangSmith API
    client = Client()
    projects = list(client.list_projects())
    print(f"\n✅ SUCCESS: Connected to LangSmith API! Found {len(projects)} projects.")
    
    # Second test: Generate a traced LLM call
    print("\nExecuting traced LLM call...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = llm.invoke([HumanMessage(content="Hello! This is a test trace. Please reply with 'Trace successful'.")])
    print(f"LLM Response: {response.content}")
    print("\n✅ Trace successfully created in background. Check your LangSmith dashboard!")
    
except Exception as e:
    print("\n❌ FAILED to connect or trace to LangSmith.")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")
