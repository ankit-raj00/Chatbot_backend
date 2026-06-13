import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

async def test_gemini_tokens():
    print("Testing Gemini Token Output...")
    
    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GOOGLE_API_KEY")
    )
    
    print("\n--- TEST 1: Simple Invoke ---")
    response = await llm.ainvoke([HumanMessage(content="Write a 3-word poem.")])
    print("Response usage_metadata:")
    print(response.usage_metadata)
    
    print("\n--- TEST 2: astream_events (what we use in the app) ---")
    
    # Create a simple graph or just stream events from the model itself
    events = llm.astream_events([HumanMessage(content="Write a 3-word poem about space.")], version="v1")
    
    async for event in events:
        event_type = event.get("event")
        if event_type == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            print("\nFound on_chat_model_end!")
            if hasattr(output, "usage_metadata"):
                print("usage_metadata from event:", output.usage_metadata)
            else:
                print("No usage_metadata attribute on output")
                print("Output object dir:", dir(output))

if __name__ == "__main__":
    asyncio.run(test_gemini_tokens())
