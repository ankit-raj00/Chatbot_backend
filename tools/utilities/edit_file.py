"""
edit_file — surgical text replacement on an existing sandbox file.

WHY: without this, the only way to fix one line in a generated file (a LaTeX
compile error, a typo in a script) is to have the LLM re-emit the file's ENTIRE
content via run_python — expensive in tokens, slower, and error-prone on large
files. This tool mirrors Claude Code's own Edit tool: give it the exact text to
find and the text to replace it with; it fails loudly if the match isn't unique
rather than guessing.
"""
import difflib
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils.workspace import workspace_for, is_path_within_sandbox


def _count_diff_lines(old_text: str, new_text: str) -> tuple[int, int]:
    """Return (added, removed) line counts via a unified diff."""
    diff = difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), lineterm="")
    added = removed = 0
    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def make_edit_file_tool(user_id: str):
    ws_root = workspace_for(user_id)

    class EditFileInput(BaseModel):
        path: str = Field(description="Sandbox-relative path to an existing file, e.g. 'work/notes.tex'")
        old_string: str = Field(description="The exact text to find. Must match exactly once in the file "
                                             "unless replace_all is true — include enough surrounding context "
                                             "(e.g. a few lines) to make the match unique.")
        new_string: str = Field(description="The text to replace old_string with.")
        replace_all: bool = Field(default=False, description="Replace every occurrence instead of requiring exactly one.")

    @tool(args_schema=EditFileInput)
    async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """
        Make a targeted edit to an existing file WITHOUT rewriting its whole
        content. Prefer this over run_python for fixing/adjusting a file you
        (or the user) already created — it's far cheaper and less error-prone
        than re-emitting the entire file.

        The file must already exist. old_string must match the file's current
        content exactly (whitespace included) — read the file first if unsure
        of its exact current text.
        """
        if not old_string:
            return "Error: old_string must not be empty."
        if old_string == new_string:
            return "Error: old_string and new_string are identical — nothing to change."

        target = (ws_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if not is_path_within_sandbox(user_id, path):
            return "BLOCKED: path outside sandbox"
        if not target.exists():
            return f"Error: file not found: {path}"
        if not target.is_file():
            return f"Error: not a file: {path}"

        try:
            original = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: {path} is not a UTF-8 text file (binary files can't be edited this way)."

        occurrences = original.count(old_string)
        if occurrences == 0:
            return (f"Error: old_string not found in {path}. "
                    "Make sure it matches the file's current content exactly, "
                    "including whitespace/indentation.")
        if occurrences > 1 and not replace_all:
            return (f"Error: old_string matches {occurrences} locations in {path}. "
                    "Include more surrounding context to make it unique, or pass replace_all=true.")

        updated = original.replace(old_string, new_string) if replace_all else original.replace(old_string, new_string, 1)

        target.write_text(updated, encoding="utf-8")
        added, removed = _count_diff_lines(original, updated)

        return f"Edited {path} (+{added} -{removed} lines)"

    return edit_file
