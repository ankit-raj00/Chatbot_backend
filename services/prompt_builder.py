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
        """Historically built a "You are AgentX..." identity + tools + generic
        instructions block. Removed (2026-08-09): this SystemMessage is always
        merged with agent_node.py's AGENT_SYSTEM_PROMPT (see its
        _with_system_prompt), which already opens with its own identity
        statement and covers search_knowledge_base far more correctly and
        completely (including the "knowledge-base documents are NOT sandbox
        files" distinction this block never had). Measured: ~148 tokens of
        pure duplication, sent on every single LLM call within every turn
        (agent_node runs once per ReAct round-trip, not once per turn) of
        every conversation — cut with zero loss, verified against the same
        test suite this function's callers are covered by."""
        return ""

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
        Build the "What I know about you" section injected into the system prompt
        from the user's stored long-term memories. Returns "" when there are none.
        """
        if not memories:
            return ""
        section = "### What I know about you\n"
        for mem in memories:
            section += f"- {mem.get('topic', '')}: {mem.get('content', '')}\n"
        return section + "\n"

    @staticmethod
    def build_conversation_summary_section(summary: str) -> str:
        """Build the "Earlier in this conversation" section from
        ConversationSummaryService's running summary — covers messages that
        have aged out of HistoryService's most-recent-window and would
        otherwise be invisible to the agent. Returns "" when there's none yet
        (short conversations never generate one)."""
        if not summary:
            return ""
        return f"### Earlier in this conversation (summarized — some detail was condensed)\n{summary}\n\n"

    @classmethod
    def assemble(
        cls,
        enabled_tools: list[str],
        mcp_resources: list[dict] = None,
        mcp_prompts: list[dict] = None,
        user_memories: list[dict] = None,
        conversation_summary: str = "",
    ) -> str:
        """Assemble the complete system prompt from all sections."""
        core    = cls.build_core_system_prompt(enabled_tools)
        mcp     = cls.build_mcp_context_section(mcp_resources or [], mcp_prompts or [])
        memory  = cls.build_memory_section(user_memories or [])
        summary = cls.build_conversation_summary_section(conversation_summary)
        return f"{core}\n{summary}{memory}{mcp}".strip()
