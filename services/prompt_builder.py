"""
PromptBuilder — constructs the system prompt for each chat turn.

Previously this was 60 lines of inline string building inside chat_controller.py.
Moving it here makes it testable and reusable.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PromptBuilder:

    @staticmethod
    def build_core_system_prompt(enabled_tools: list[str]) -> str:
        """Build the core system prompt based on which tools are enabled."""
        tools_desc = ""

        if "search_knowledge_base" in enabled_tools:
            tools_desc += (
                "- **search_knowledge_base**: Search file contents the user has selected. "
                "ALWAYS use this if the user asks about their files, resume, or documents.\n"
                "- **read_document_page**: Read a specific page if search results are truncated.\n"
            )

        if "list_google_drive_folders" in enabled_tools or "create_google_drive_folder" in enabled_tools:
            tools_desc += "- Manage Drive files using the Google Drive tools.\n"

        core = (
            "You are AgentX, a powerful AI assistant with access to tools, a Knowledge Base, and memory.\n\n"
        )
        if tools_desc:
            core += f"### YOUR TOOLS:\n{tools_desc}\n"

        core += (
            "### INSTRUCTIONS:\n"
            "1. Always check your available tools before refusing a request.\n"
            "2. If the user selects context files, use `search_knowledge_base` first.\n"
            "3. Combine tool outputs for comprehensive answers.\n"
            "4. If you encounter an error, explain it clearly.\n"
        )
        return core

    @staticmethod
    def build_mcp_context_section(
        available_resources: list[dict],
        available_prompts: list[dict]
    ) -> str:
        """Build the MCP resources/prompts section of the system prompt."""
        section = ""

        if available_resources:
            section += "### Available MCP Context Resources\n"
            section += "Use `read_mcp_resource` to read these if needed:\n"
            for r in available_resources:
                section += f"- **{r['name']}** ({r['mimeType']})\n  URI: `{r['uri']}`\n  Description: {r['description']}\n"
            section += "\n"

        if available_prompts:
            section += "### Available MCP Prompts\n"
            for p in available_prompts:
                args_str = ", ".join(arg['name'] for arg in p.get('arguments', []))
                section += f"- **{p['name']}**: {p['description']}\n  Arguments: {args_str}\n"
            section += "\n"

        return section

    @staticmethod
    def build_memory_section(memories: list[dict]) -> str:
        """
        Build the memory section injected into the system prompt.
        Phase 8 populates this. Returns empty string until Phase 8 is implemented.
        """
        if not memories:
            return ""
        section = "### What I know about you\n"
        for mem in memories:
            section += f"- {mem.get('topic', '')}: {mem.get('content', '')}\n"
        return section + "\n"

    @classmethod
    def assemble(
        cls,
        enabled_tools: list[str],
        mcp_resources: list[dict] = None,
        mcp_prompts: list[dict] = None,
        user_memories: list[dict] = None,
    ) -> str:
        """Assemble the complete system prompt from all sections."""
        core = cls.build_core_system_prompt(enabled_tools)
        mcp = cls.build_mcp_context_section(
            mcp_resources or [],
            mcp_prompts or []
        )
        memory = cls.build_memory_section(user_memories or [])
        return f"{core}\n{memory}{mcp}".strip()
