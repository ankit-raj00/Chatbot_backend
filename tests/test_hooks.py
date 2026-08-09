"""
Tests for utils/hooks.py's centralized sandbox guardrails (Tier 2.3).

The whole point of centralizing these into a registered pre-tool hook is
that a NEW tool automatically inherits protection through agent_tool_node's
dispatch, even if its author never calls check_sandbox_path/
check_run_shell_command directly. These tests prove that path specifically —
going through run_pre_tool_hooks() itself, not through any tool's own
function body — which is what the existing per-tool tests
(test_agent_v3.py, test_e2e.py) don't cover.
"""
import pytest

from utils.hooks import (
    run_pre_tool_hooks, check_sandbox_path, check_run_shell_command, BLOCKED_PATTERNS,
)

USER_ID = "test_hooks_user"
CONV_ID = "test_hooks_conv"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,path_arg,bad_path", [
    ("sandbox_edit_file", "path", "../../etc/passwd"),
    ("sandbox_analyze_image", "sandbox_path", "../../../etc/shadow"),
    ("sandbox_run_python", "filename", "../outside.py"),
])
async def test_hook_denies_path_escape_without_the_tool_doing_anything(tool_name, path_arg, bad_path):
    """The hook alone must catch this — not the tool's own code, which this
    call never touches at all."""
    result = await run_pre_tool_hooks(tool_name, {path_arg: bad_path}, USER_ID, CONV_ID)
    assert result is not None and result.get("deny") is True
    assert "outside sandbox" in result["reason"]


@pytest.mark.asyncio
async def test_hook_allows_a_safe_path():
    result = await run_pre_tool_hooks("sandbox_edit_file", {"path": "work/notes.txt"}, USER_ID, CONV_ID)
    assert result is None


@pytest.mark.asyncio
async def test_hook_denies_run_shell_escape_via_hook_alone():
    result = await run_pre_tool_hooks("sandbox_run_shell", {"command": "cat ../../etc/passwd"}, USER_ID, CONV_ID)
    assert result is not None and result.get("deny") is True


@pytest.mark.asyncio
async def test_hook_denies_run_shell_destructive_pattern():
    result = await run_pre_tool_hooks("sandbox_run_shell", {"command": "rm -rf /"}, USER_ID, CONV_ID)
    assert result is not None and result.get("deny") is True


@pytest.mark.asyncio
async def test_hook_allows_a_safe_shell_command():
    result = await run_pre_tool_hooks("sandbox_run_shell", {"command": "echo hello"}, USER_ID, CONV_ID)
    assert result is None


@pytest.mark.asyncio
async def test_hook_ignores_unrelated_tools():
    """A tool with no sandbox-path argument at all (e.g. list_skills) must
    never be denied by this guard."""
    result = await run_pre_tool_hooks("list_skills", {}, USER_ID, CONV_ID)
    assert result is None


def test_check_sandbox_path_returns_none_for_unmapped_tool():
    """A tool this guard doesn't know about (not in _SANDBOX_PATH_ARG_BY_TOOL)
    is correctly left alone by this specific check — that's not a gap, it's
    tools with no sandbox-path argument to validate in the first place."""
    assert check_sandbox_path("some_future_tool", {"anything": "x"}, USER_ID, CONV_ID) is None


def test_check_run_shell_command_matches_every_blocked_pattern():
    for pattern in BLOCKED_PATTERNS:
        assert check_run_shell_command(f"echo before && {pattern} && echo after",
                                       USER_ID, CONV_ID) is not None, f"{pattern!r} should be blocked"
