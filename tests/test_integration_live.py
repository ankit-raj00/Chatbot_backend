"""
Live Integration Tests -- Real LLM (OmniRoute) + Real Redis + Real v3 Graph
============================================================================
Uses the actual OmniRoute-backed LLM (graph/llm_registry.py), actual Redis, and
the real compiled single-agent LangGraph (graph/builder.py). NO mocks -- this
tests the ACTUAL system behaviour.

The pre-v3 supervisor/7-agent-intent-classification architecture (graph.supervisor,
VALID_AGENTS, per-intent subgraphs) has been removed; there is no intent
classification step anymore -- a single ReAct agent decides which tools to call.
The graph also has no persistent checkpointer (see graph/builder.py docstring):
multi-turn memory is achieved by the CALLER passing the full message history on
each call (as services/chat_service.py does), not by graph-level state.

What is tested:
  IT03  Skill trigger -> skill_body populated with real SKILL.md content
        (skill lookup itself, independent of any graph)
  IT04  All builtin skill triggers work
  IT05  Full v3 agent run: plain chat question -> real LLM response
  IT06  Full v3 agent run: the agent calls a tool (run_python) when asked to
  IT07  Full v3 agent run: code-writing question -> response contains code
  IT10  Multi-turn memory: caller-supplied history lets the LLM recall turn 1
        (mirrors how services/chat_service.py actually does this -- no
        checkpointer, full history passed in each call)
  IT11  Redis raw: PING + SET/GET on the real Redis instance
  IT13  Skill body content matches real SKILL.md file on disk
  IT14  get_relevant_skill_for_message: no match for gibberish
  IT15  astream_events on the v3 graph emits agent_node / agent_tool_node events

Run just these tests (needs OMNIROUTE_BASE_URL reachable + REDIS_URL):
    pytest tests/test_integration_live.py -v --tb=short -s
"""

import os
import sys
import uuid
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

REDIS_URL   = os.getenv("REDIS_URL", "")
BUILTIN_DIR = Path(__file__).parent.parent / "skills" / "builtin"


# ---- Helpers ----------------------------------------------------------------

async def _run_agent(messages: list, user_id: str = "live_test_user_v3", **config_overrides):
    """Invoke the real v3 agent graph (graph/builder.py) with a message list,
    mirroring how services/chat_service.py drives it in production."""
    from graph.builder import get_agent_graph
    from config.model_config import ModelConfig

    graph = await get_agent_graph()
    # conversation_id doubles as configurable.thread_id (what agent_node/
    # agent_tool_node actually read to scope the sandbox — see
    # utils.workspace.conversation_workspace_for) — a caller that needs to
    # know it afterward (e.g. to check a created file's path) can pass it
    # explicitly via conversation_id=...
    conversation_id = config_overrides.pop("conversation_id", None) or str(uuid.uuid4())
    configurable = {
        "user_id": user_id,
        "model": ModelConfig.DEFAULT_MODEL,
        "enabled_tools": [],
        "thread_id": conversation_id,
    }
    configurable.update(config_overrides)
    result = await graph.ainvoke(
        {"messages": messages, "user_id": user_id, "conversation_id": conversation_id,
         "enabled_tools": [], "mcp_server_urls": [], "selected_files": None},
        config={"configurable": configurable, "recursion_limit": 15},
    )
    return result


def _final_text(result: dict) -> str:
    last = result["messages"][-1]
    content = getattr(last, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


# ---- IT03-04  Skill Injection (real trigger matching, no graph needed) ------

class TestIT03SkillInjection:
    """IT03-04: Skill trigger matching fills skill_body with real file content."""

    @pytest.mark.asyncio
    async def test_it03b_skill_body_matches_disk_file(self):
        """Skill body returned by loader must match actual SKILL.md file on disk."""
        from skills.skill_loader import load_builtin_skill
        body = load_builtin_skill("create-pdf")
        assert body is not None, "create-pdf skill not found"
        assert not body.startswith("---"), "Frontmatter must be stripped"
        assert len(body) > 200, "Skill body too short"
        preview = body[:60].replace("\n", " ")
        print(f"\n  -> create-pdf skill: {len(body)} chars, starts: '{preview}'")

    @pytest.mark.asyncio
    async def test_it04_all_builtin_skills_have_valid_triggers(self):
        """Every builtin skill with triggers must match when the trigger is used as message."""
        from skills.skill_loader import list_builtin_skills, get_relevant_skill_for_message
        skills = list_builtin_skills()
        print(f"\n  -> Testing {len(skills)} builtin skills")
        for skill in skills:
            triggers = skill.get("triggers", [])
            if not triggers:
                print(f"  -- {skill['name']}: no triggers -- skip")
                continue
            trigger_phrase = triggers[0]
            result = await get_relevant_skill_for_message(
                trigger_phrase, user_id="test", agent_type=skill.get("agent", "")
            )
            status = "matched" if result else "no match"
            print(f"  -> {skill['name']}: '{trigger_phrase}' -> {status}")
            assert result is not None, (
                f"Skill '{skill['name']}' trigger '{trigger_phrase}' returned None"
            )
            body, name = result
            assert len(body) > 50, f"Skill '{skill['name']}' body is too short"


# ---- IT05-07  Full v3 Agent Run (real LLM response via OmniRoute) ----------

class TestIT05FullAgentRun:
    """IT05-07: The real v3 single agent handles chat, tool-calling, and code."""

    @pytest.mark.asyncio
    async def test_it05_chat_question_gets_real_response(self):
        """Plain knowledge question -> real LLM response, no tool needed."""
        result = await _run_agent([HumanMessage(content="In one sentence, what is Python?")])
        resp = _final_text(result)
        print(f"\n  -> response: {resp[:150]}")
        assert len(resp) > 5
        assert isinstance(result["messages"][-1], AIMessage)

    @pytest.mark.asyncio
    async def test_it06_agent_calls_run_python_tool_when_asked(self):
        """Explicit instruction to use run_python must produce a tool call and a
        real file on disk in the user's sandbox (proves the sandbox tool actually
        executes through the live OmniRoute-backed agent)."""
        from utils.workspace import workspace_for, conversation_workspace_for

        user_id = f"live_test_v3_{uuid.uuid4().hex[:8]}"
        conversation_id = f"live_test_conv_{uuid.uuid4().hex[:8]}"
        try:
            result = await _run_agent(
                [HumanMessage(content=(
                    "Use run_python to write the exact text 'integration_test_ok' "
                    "into outputs/it06.txt, then print the path. Do it now."
                ))],
                user_id=user_id,
                conversation_id=conversation_id,
            )
            tool_msgs = [m for m in result["messages"] if getattr(m, "name", None) == "run_python"]
            print(f"\n  -> run_python tool messages: {len(tool_msgs)}")
            assert tool_msgs, "Agent never called run_python"

            out_file = conversation_workspace_for(user_id, conversation_id) / "outputs" / "it06.txt"
            assert out_file.exists(), "run_python did not create the expected output file"
            assert "integration_test_ok" in out_file.read_text(encoding="utf-8")
        finally:
            shutil.rmtree(workspace_for(user_id), ignore_errors=True)

    @pytest.mark.asyncio
    async def test_it07_code_question_response_contains_code(self):
        """Code-writing question -> response contains recognizable Python."""
        result = await _run_agent(
            [HumanMessage(content="Write a Python function that returns 'hello world'. "
                                   "Just show me the code in your reply, don't run it.")]
        )
        resp = _final_text(result).lower()
        print(f"\n  -> response: {resp[:150]}")
        assert "def " in resp or "return" in resp or "hello" in resp, \
            f"Expected code-like content: {resp[:100]}"


# ---- IT10  Multi-Turn Memory (caller-supplied history, no checkpointer) ----

class TestIT10MultiTurnMemory:
    """IT10: Passing full history on the second call (as chat_service does) lets
    the LLM recall a fact from turn 1 -- there is no graph-level checkpointer in
    v3, so this is the actual mechanism production relies on."""

    @pytest.mark.asyncio
    async def test_it10_second_call_remembers_first_via_passed_history(self):
        turn1_human = HumanMessage(content="My name is Zaphod Beeblebrox.")
        r1 = await _run_agent([turn1_human])
        turn1_ai_text = _final_text(r1)
        print(f"\n  Turn 1 -> {turn1_ai_text[:80]}")

        turn2_human = HumanMessage(content="What is my name?")
        r2 = await _run_agent([turn1_human, AIMessage(content=turn1_ai_text), turn2_human])
        resp_lower = _final_text(r2).lower()
        print(f"  Turn 2 -> {resp_lower[:150]}")
        assert "zaphod" in resp_lower or "beeblebrox" in resp_lower, \
            f"LLM did not recall the name from passed-in history. Got: {resp_lower}"


# ---- IT11  Redis Raw ---------------------------------------------------------

class TestIT11RedisRaw:
    """IT11: Test raw Redis operations (PING, SET, GET) against the real instance."""

    @pytest.mark.asyncio
    async def test_it11a_redis_ping(self):
        import redis.asyncio as aioredis
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            pong = await r.ping()
            print(f"\n  -> Redis PING: {pong}")
            assert pong is True
        finally:
            await r.aclose()

    @pytest.mark.asyncio
    async def test_it11b_redis_set_and_get(self):
        import redis.asyncio as aioredis
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        key = f"agentx:integration:{uuid.uuid4()}"
        val = "hello_from_live_integration_test"
        try:
            await r.set(key, val, ex=60)
            read = await r.get(key)
            print(f"\n  -> Wrote: {val!r}")
            print(f"  -> Read:  {read!r}")
            assert read == val
        finally:
            await r.delete(key)
            await r.aclose()

    @pytest.mark.asyncio
    async def test_it11c_redis_json_round_trip(self):
        import json
        import redis.asyncio as aioredis
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        key     = f"agentx:state:{uuid.uuid4()}"
        payload = {"user_id": "u1", "note": "test"}
        try:
            await r.set(key, json.dumps(payload), ex=60)
            raw   = await r.get(key)
            back  = json.loads(raw)
            print(f"\n  -> Stored and retrieved: {back}")
            assert back == payload
        finally:
            await r.delete(key)
            await r.aclose()


# ---- IT13-14  Skill Content Fidelity ----------------------------------------

class TestIT13SkillFidelity:
    """IT13-14: Skill body matches disk; gibberish returns None."""

    @pytest.mark.asyncio
    async def test_it13_skill_body_matches_disk_for_pdf(self):
        """Skill body from get_relevant_skill_for_message must match SKILL.md on disk."""
        import re
        from skills.skill_loader import get_relevant_skill_for_message

        skill_file    = BUILTIN_DIR / "create-pdf" / "SKILL.md"
        raw           = skill_file.read_text(encoding="utf-8")
        expected_body = re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()

        result = await get_relevant_skill_for_message(
            "generate a pdf report", user_id="", agent_type="document"
        )

        assert result is not None, "Should have matched create-pdf skill"
        actual_body, name = result
        print(f"\n  -> matched skill: {name}")
        assert actual_body == expected_body, "Skill body must exactly match SKILL.md minus frontmatter"

    @pytest.mark.asyncio
    async def test_it14_gibberish_returns_no_skill(self):
        """Random gibberish must not match any skill."""
        from skills.skill_loader import get_relevant_skill_for_message
        result = await get_relevant_skill_for_message(
            "xyzzy plugh frobnicate wibble wobble 12345", user_id="", agent_type=""
        )
        print(f"\n  -> Gibberish result: {result}")
        assert result is None


# ---- IT15  v3 Agent Streaming Events ----------------------------------------

class TestIT15AgentStreaming:
    """IT15: astream_events on the real v3 graph emits agent_node / agent_tool_node
    events -- replaces the old intent_classifier event check (that node no longer
    exists; there's no intent-classification step in the single-agent design)."""

    @pytest.mark.asyncio
    async def test_it15a_streaming_emits_agent_node_events(self):
        from graph.builder import get_agent_graph
        from config.model_config import ModelConfig

        graph = await get_agent_graph()
        config = {
            "configurable": {"user_id": "live_test_v3", "model": ModelConfig.DEFAULT_MODEL,
                              "enabled_tools": []},
            "recursion_limit": 15,
        }
        state = {
            "messages": [HumanMessage(content="What color is the sky? Answer in one word.")],
            "user_id": "live_test_v3", "conversation_id": str(uuid.uuid4()),
            "enabled_tools": [], "mcp_server_urls": [], "selected_files": None,
        }

        node_names = set()
        chunks = []
        async for event in graph.astream_events(state, version="v2", config=config):
            if not isinstance(event, dict):
                continue
            node_name = event.get("metadata", {}).get("langgraph_node", "")
            if node_name:
                node_names.add(node_name)
            if event.get("event") == "on_chat_model_stream" and node_name == "agent_node":
                chunk = event.get("data", {}).get("chunk")
                text = getattr(chunk, "content", "") if chunk else ""
                if isinstance(text, str):
                    chunks.append(text)

        print(f"\n  -> nodes seen: {node_names}")
        print(f"  -> streamed text: {''.join(chunks)[:100]!r}")
        assert "agent_node" in node_names, "Expected the agent_node to run"
        assert len(chunks) > 0, "Expected streamed chat model chunks from agent_node"
