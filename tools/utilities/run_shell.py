"""
run_shell — sandboxed shell command execution, bound as a real tool.

Now supports live streaming via `stream_shell`, path validation,
and per-user npm/pip environment scoping.
"""
import os
from langchain_core.tools import tool
from langchain_core.callbacks import adispatch_custom_event
from utils.workspace import conversation_workspace_for, is_path_within_conversation_sandbox, pip_cache_dir_for, npm_prefix_for
from utils.code_executor import stream_shell, sandbox_env
from utils import sandbox_client

BLOCKED_PATTERNS = [
    "rm -rf /", "rm -rf ~", "sudo rm", ":(){:|:&};:", "mkfs",
    "dd if=/dev/zero", "chmod -R 777 /", "> /dev/sda",
    "curl | sh", "wget | sh", "curl | bash", "wget | bash",
    # Confirmed live: with NO restriction at all, a benign prompt led the
    # agent to reach loopback (enumerating this app's own /admin/* routes)
    # and the AWS metadata endpoint from inside run_python — this is the
    # same protection for raw shell commands (curl/wget/nc/etc.), which the
    # Python-level socket guard in code_executor.py doesn't cover since
    # they're not Python code. Simple substring match (same mechanism as
    # every other entry above), so this can over-match on an unrelated
    # command that happens to mention one of these strings (e.g. grepping a
    # log file containing "127.0.0.1") — an accepted tradeoff for a coding
    # sandbox where the agent can always take a different approach if
    # legitimately blocked. Broader ranges (RFC1918 generally) are enforced
    # instead at the host firewall and the Python guard, where over-matching
    # on short numeric substrings isn't a risk.
    "169.254.169.254", "169.254.", "127.0.0.1", "localhost", "172.17.",
]


def make_run_shell_tool(user_id: str, conversation_id: str):
    cwd = str(conversation_workspace_for(user_id, conversation_id))

    @tool
    async def run_shell(command: str) -> str:
        """
        Execute a shell command inside the user's sandboxed workspace
        directory and return combined stdout+stderr.

        All operations happen in a sandboxed per-user directory — you
        cannot access system directories. Destructive commands
        (rm -rf, sudo, fork bombs, curl|sh, etc.) are blocked.
        Path traversals (e.g. ../../otheruser) are blocked.

        Use for: listing/inspecting files, running scripts you've already
        written, checking command output, exploring the workspace.

        Args:
            command: The shell command to execute.
        """
        import shlex

        # Tokenize with shell OPERATORS (; & | < >) split into their own tokens,
        # so `cd ..; ls` yields ['cd', '..', ';', 'ls'] and the '..' is caught,
        # rather than a glued '..;' token that would slip past the checks below.
        try:
            lex = shlex.shlex(command, posix=True, punctuation_chars=True)
            lex.whitespace_split = True
            parts = list(lex)
        except ValueError:
            parts = command.split()

        def _norm(p: str) -> str:
            return p.replace("\\", "/")

        for i, part in enumerate(parts):
            p = _norm(part)
            # Block parent-directory traversal even when the token has no other
            # slash (e.g. bare `cd ..`), which the slash-only check used to miss.
            if p == ".." or p.startswith("../") or "/../" in p or p.endswith("/.."):
                return "BLOCKED: parent-directory ('..') access is not allowed"
            # Block home-dir / absolute escapes.
            if p.startswith("~"):
                return "BLOCKED: home-directory ('~') access is not allowed"
            if ("/" in p) and not is_path_within_conversation_sandbox(user_id, conversation_id, part):
                return "BLOCKED: path outside sandbox"
            # Guard the `cd`/`pushd`/`chdir` target explicitly (it mutates cwd for
            # every following command in the same shell invocation).
            if p.lower() in ("cd", "pushd", "chdir") and i + 1 < len(parts):
                target = parts[i + 1]
                tnorm = _norm(target)
                if (tnorm == ".." or tnorm.startswith("../") or "/../" in tnorm
                        or tnorm.startswith("~")
                        or not is_path_within_conversation_sandbox(user_id, conversation_id, target)):
                    return "BLOCKED: cannot change directory outside the sandbox"

        if sandbox_client.is_remote():
            # The blocked-pattern check still runs here (above/below) rather than
            # only on SES: it returns a clear message without paying a network
            # round-trip, and the container's own limits remain the real control.
            cl = command.lower()
            if any(b in cl for b in BLOCKED_PATTERNS):
                return "BLOCKED: command contains a forbidden pattern"
            stream = sandbox_client.stream_shell_remote(user_id, conversation_id, command, 120)
        else:
            # Prepare isolated environment. SECURITY: build off sandbox_env()'s safe
            # allowlist, never {**os.environ} — the backend's real environment holds
            # every service credential (QDRANT_API_KEY, MONGO_URI, GOOGLE_API_KEY,
            # JWT_SECRET_KEY, ...) and sandboxed user code must never be able to read
            # them (this was a confirmed exploitable leak — see code_executor.py).
            npm_prefix = npm_prefix_for(user_id)
            env = sandbox_env({
                "PIP_CACHE_DIR": str(pip_cache_dir_for(user_id)),
                "NPM_CONFIG_PREFIX": str(npm_prefix),
            })
            env["PATH"] = f"{npm_prefix / 'bin'}{os.pathsep}{env.get('PATH', '')}"
            stream = stream_shell(command, cwd, timeout=120,
                                  blocked_patterns=BLOCKED_PATTERNS, env=env)

        lines = []
        async for item in stream:
            if "line" in item:
                lines.append(item["line"])
                await adispatch_custom_event(
                    "exec_output",
                    {"tool": "run_shell", "line": item["line"], "stream": item["stream"]},
                )
                
        return "\n".join(lines) or "(no output)"

    return run_shell
