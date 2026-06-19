import asyncio
import os
from dotenv import load_dotenv

from core.state import State
from graph.builder import create_chat_graph
from langchain_core.messages import HumanMessage

load_dotenv()

async def test_graph_tokens():
    print("Testing LangGraph Token Output...")
    
    # Initialize the graph
    graph = create_chat_graph()
    
    # Setup simple state
    state = {
        "messages": [HumanMessage(content="Write a 3-word poem about space.")],
        "user_id": "test_user",
        "current_model": "gemini-2.5-flash"
    }
    config = {
        "configurable": {
            "enabled_tools": [],
            "user_id": "test_user",
            "model": "gemini-2.5-flash"
        }
    }
    
    print("\n--- TEST: astream_events with LangGraph ---")
    events = graph.astream_events(state, version="v1", config=config)
    
    async for event in events:
        event_type = event.get("event")
        name = event.get("name")
        
        # Print all end events to see what we get
        if event_type.endswith("_end"):
            print(f"Event: {event_type} | Name: {name}")
            if event_type == "on_chat_model_end":
                output = event.get("data", {}).get("output")
                print(f"-> Found on_chat_model_end!")
                if hasattr(output, "usage_metadata"):
                    print(f"-> usage_metadata: {output.usage_metadata}")
                else:
                    print(f"-> No usage_metadata!")
                    
if __name__ == "__main__":
    asyncio.run(test_graph_tokens())
