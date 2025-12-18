
import os
import sys

def cleanup():
    # List of files to delete (Relative to backend/)
    # Ensure this script is run from backend/ or we adjust paths.
    # Assuming run from D:\Gemini Playgroun\vscodeground\chatbot
    
    base_dir = r"D:\Gemini Playgroun\vscodeground\chatbot\backend"
    
    files_to_delete = [
        "controllers/mcp_controller.py",         # Replaced by mcp_server_controller.py
        "controllers/chat_controller_stream.py", # Replaced by LangGraph controller
        "services/mcp_server.py",                # Legacy FastMCP server
        "services/google_drive_server.py",       # Legacy FastMCP server
        "utils/langchain_tools.py",              # Legacy bridge
        "graph/chat_agent.py",                   # Legacy agent
        "check_mcp_db.py",                       # One-off script
        "cleanup_mcp_duplicates.py",             # One-off script
        "fix_mcp_duplicates.py",                 # One-off script
        "migrate_to_native_tools.py",            # One-off script
        "test_client_methods.py",                # Temporary test
        "test_mcp_connection.py"                 # Temporary test
    ]

    print("Starting Cleanup of Legacy Files...")
    print(f"Base Directory: {base_dir}")
    
    deleted_count = 0
    
    for relative_path in files_to_delete:
        full_path = os.path.join(base_dir, relative_path)
        
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                print(f"[DELETED] {relative_path}")
                deleted_count += 1
            except Exception as e:
                print(f"[ERROR] Could not delete {relative_path}: {e}")
        else:
            print(f"[SKIPPED] {relative_path} (Not found)")

    print("-" * 30)
    print(f"Cleanup Complete. Deleted {deleted_count} files.")

if __name__ == "__main__":
    cleanup()
