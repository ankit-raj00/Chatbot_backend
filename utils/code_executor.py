"""
Shared Python code execution utility for all agent subgraphs.
Provides sandboxed subprocess execution with auto-install support.
"""
import asyncio
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# ── Guardrails ──────────────────────────────────────────────────────────────
MAX_LINE_LEN      = 8_192       # truncate any single output line
MAX_TOTAL_OUTPUT  = 2_000_000   # ~2 MB hard cap on streamed output (bounds memory)
_REAP_TIMEOUT     = 5           # seconds to wait for a killed process to die

# ── Sandbox environment isolation ────────────────────────────────────────────
# CRITICAL: subprocess.Popen/create_subprocess_exec inherit the FULL parent
# environment when `env=` is omitted. The backend process's environment holds
# every service credential (QDRANT_API_KEY, MONGO_URI, GOOGLE_API_KEY,
# JWT_SECRET_KEY, OMNIROUTE_API_KEY, CLOUDINARY_*, REDIS_URL, LLAMA_CLOUD_API_KEY,
# TAVILY_API_KEY, ...). User-authored code executed here (run_python/run_shell)
# must NEVER be able to read those via os.environ — doing so lets any user read
# the shared Qdrant/Mongo credentials directly and bypass all tenancy filtering
# (proven exploitable: a sandboxed script imported qdrant-client, read the
# inherited QDRANT_URL/QDRANT_API_KEY, and scrolled OTHER users' vectors).
# Every subprocess spawned in this module MUST pass env=sandbox_env(...).
_SAFE_ENV_PASSTHROUGH = {
    "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "APPDATA",
    "LOCALAPPDATA", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE", "HOME", "LANG", "LC_ALL",
    "PYTHONIOENCODING", "PYTHONUTF8", "PYTHONUNBUFFERED",
}


def sandbox_env(extra: dict | None = None) -> dict:
    """Minimal, secret-free environment for a sandboxed subprocess. `extra`
    layers sandbox-scoping vars (PIP_CACHE_DIR, NPM_CONFIG_PREFIX, ...) on top
    — never pass anything from a full os.environ copy as `extra`."""
    base = {k: v for k, v in os.environ.items() if k.upper() in _SAFE_ENV_PASSTHROUGH}
    if extra:
        base.update(extra)
    return base


# ── Filesystem guard for run_python ──────────────────────────────────────────
# run_python executes arbitrary user-authored Python. `cwd=` only sets the
# DEFAULT for relative paths — it does NOT stop the script from using an
# absolute path or `..` to read/write anywhere the OS user can (proven: a
# script used python-docx/zipfile, which calls the builtin open(), to read a
# file from the operator's OneDrive folder, well outside the workspace).
#
# This patches open()/os.open()/os.scandir()/os.listdir() (which also covers
# pathlib.Path.open/read_text/write_text — they call the same io.open — and
# os.walk, which is built on os.scandir) to reject any path that resolves
# outside the workspace root.
#
# ⚠️ HONEST LIMITATION: this is an in-process Python-level guard, not OS-level
# isolation. It stops the class of accidental/exploratory escape observed here
# (direct file reads via open()/pathlib/python-docx etc). It does NOT stop a
# determined attacker who shells out via subprocess to a raw OS command, or
# calls into a C extension via ctypes to bypass Python's file APIs entirely.
# The robust fix is OS/container-level isolation (a dedicated low-privilege
# OS account restricted by real filesystem ACLs, or a container per execution)
# — this guard meaningfully raises the bar today without that larger project.
_SANDBOX_GUARD_PREAMBLE = (
    "import os as _os, builtins as _b, pathlib as _pl\n"
    "_ROOT = _os.path.abspath(_os.getcwd())\n"
    "def _sbx_check(_p):\n"
    "    try:\n"
    "        _ap = _os.path.abspath(_os.fspath(_p))\n"
    "    except Exception:\n"
    "        return\n"
    "    if not (_ap == _ROOT or _ap.startswith(_ROOT + _os.sep)):\n"
    "        raise PermissionError('SANDBOX: access outside the workspace is not allowed: ' + repr(_p))\n"
    "_real_open = _b.open\n"
    "def _sbx_open(_f, *a, **k):\n"
    "    if isinstance(_f, (str, bytes, _os.PathLike)):\n"
    "        _sbx_check(_f)\n"
    "    return _real_open(_f, *a, **k)\n"
    "_b.open = _sbx_open\n"
    "_real_osopen = _os.open\n"
    "def _sbx_osopen(_p, *a, **k):\n"
    "    _sbx_check(_p)\n"
    "    return _real_osopen(_p, *a, **k)\n"
    "_os.open = _sbx_osopen\n"
    "_real_scandir = _os.scandir\n"
    "def _sbx_scandir(_p='.'):\n"
    "    _sbx_check(_p)\n"
    "    return _real_scandir(_p)\n"
    "_os.scandir = _sbx_scandir\n"
    "_real_listdir = _os.listdir\n"
    "def _sbx_listdir(_p='.'):\n"
    "    _sbx_check(_p)\n"
    "    return _real_listdir(_p)\n"
    "_os.listdir = _sbx_listdir\n"
    # pathlib.Path methods do NOT reliably route through builtins.open/os.open
    # in modern CPython (proven: Path.read_text() bypassed both patches above
    # and returned real file content) — patch the Path methods directly.
    # WindowsPath/PosixPath inherit these from Path, so patching the base
    # class alone covers both without double-wrapping.
    "for _m in ('open', 'read_text', 'read_bytes', 'write_text', 'write_bytes', 'iterdir', 'glob', 'rglob', 'unlink', 'rename', 'replace'):\n"
    "    if hasattr(_pl.Path, _m):\n"
    "        def _mk(_orig=getattr(_pl.Path, _m)):\n"
    "            def _wrap(self, *a, **k):\n"
    "                _sbx_check(self); return _orig(self, *a, **k)\n"
    "            return _wrap\n"
    "        setattr(_pl.Path, _m, _mk())\n"
)


def _guarded_launch_args(python_executable: str, script_file: str) -> list[str]:
    """
    Build the interpreter argv that installs the filesystem guard BEFORE
    executing the user's script — via -c so the saved script FILE itself
    stays byte-for-byte the user's code (still cleanly editable with
    edit_file afterward; no wrapper lines injected into it).
    """
    runner = (
        _SANDBOX_GUARD_PREAMBLE
        + f"exec(compile(open({script_file!r}, encoding='utf-8').read(), {script_file!r}, 'exec'))\n"
    )
    return [python_executable, "-u", "-c", runner]


async def _merge_async_generators(*generators):
    queue = asyncio.Queue()
    
    async def _reader(gen):
        try:
            async for item in gen:
                await queue.put(item)
        except Exception as e:
            await queue.put({"error": str(e)})
        finally:
            await queue.put(None)

    tasks = [asyncio.create_task(_reader(gen)) for gen in generators]
    active = len(tasks)
    
    while active > 0:
        item = await queue.get()
        if item is None:
            active -= 1
        else:
            yield item


async def _kill_and_reap(proc):
    """Kill a process and wait (bounded) for it to actually die, so we never leak
    a zombie or hang forever on proc.wait()."""
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT)
    except (asyncio.TimeoutError, ProcessLookupError):
        pass


async def _read_stream(stream, label):
    """Yield output lines, capping any single line so one newline-free multi-GB
    write can't buffer unboundedly."""
    while True:
        try:
            line = await stream.readline()
        except (ValueError, asyncio.LimitOverrunError):
            # Line exceeded the StreamReader buffer — drain a bounded chunk instead.
            chunk = await stream.read(MAX_LINE_LEN)
            if not chunk:
                break
            yield {"line": chunk.decode("utf-8", "replace") + " …[line truncated]", "stream": label}
            continue
        if not line:
            break
        text = line.decode("utf-8", "replace").rstrip("\n")
        if len(text) > MAX_LINE_LEN:
            text = text[:MAX_LINE_LEN] + " …[line truncated]"
        yield {"line": text, "stream": label}


async def _stream_proc(proc, timeout: int):
    """Drive a subprocess to completion, enforcing a real WALL-CLOCK deadline
    (not a per-line reset) and a total-output cap, reaping the process on exit."""
    deadline = time.monotonic() + timeout
    total = 0
    merged = _merge_async_generators(
        _read_stream(proc.stdout, "stdout"),
        _read_stream(proc.stderr, "stderr"),
    )
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await _kill_and_reap(proc)
                yield {"line": f"TIMEOUT: exceeded {timeout}s, process killed", "stream": "stderr"}
                yield {"done": True, "exit_code": -1}
                return
            try:
                item = await asyncio.wait_for(merged.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                await _kill_and_reap(proc)
                yield {"line": f"TIMEOUT: exceeded {timeout}s, process killed", "stream": "stderr"}
                yield {"done": True, "exit_code": -1}
                return

            if "line" in item:
                total += len(item["line"])
                if total > MAX_TOTAL_OUTPUT:
                    yield {"line": " …[output truncated: exceeded output limit, process killed]", "stream": "stderr"}
                    await _kill_and_reap(proc)
                    yield {"done": True, "exit_code": -1}
                    return
            yield item

        # Normal completion — reap with a bound so a stuck grandchild can't hang us.
        try:
            exit_code = await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT)
        except asyncio.TimeoutError:
            await _kill_and_reap(proc)
            exit_code = -1
        yield {"done": True, "exit_code": exit_code}
    finally:
        if proc.returncode is None:
            await _kill_and_reap(proc)


async def stream_python(code: str, cwd: str, timeout: int = 300, python_executable: str = sys.executable,
                         script_path: str | None = None, env: dict | None = None):
    """
    Like run_python, but an async generator yielding output lines as they
    arrive. Final yielded item is a dict: {"done": True, "exit_code": int}.
    Intermediate items are dicts: {"line": str, "stream": "stdout"|"stderr"}.

    script_path: if given (absolute path, inside cwd), the script is written
        there and left in place after execution — so it can be inspected and
        surgically fixed with an edit tool + re-run, instead of the caller
        having to resend the entire script every time. If omitted, the script
        is written to a throwaway temp file and deleted afterward (today's
        behavior, appropriate for one-off snippets).
    """
    Path(cwd).mkdir(parents=True, exist_ok=True)
    persist = script_path is not None
    if persist:
        Path(script_path).parent.mkdir(parents=True, exist_ok=True)
        Path(script_path).write_text(code, encoding="utf-8")
        tmp = script_path
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=cwd, delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name

    proc = await asyncio.create_subprocess_exec(
        *_guarded_launch_args(python_executable, tmp),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=sandbox_env(env),
    )
    try:
        async for item in _stream_proc(proc, timeout):
            yield item
    finally:
        if not persist:
            try:
                os.unlink(tmp)
            except OSError:
                pass


async def stream_shell(command: str, cwd: str, timeout: int = 120, blocked_patterns: list[str] | None = None, env: dict | None = None):
    """Same shape as stream_python but for shell commands."""
    if blocked_patterns:
        cl = command.lower()
        if any(b in cl for b in blocked_patterns):
            yield {"line": "BLOCKED: command contains a forbidden pattern", "stream": "stderr"}
            yield {"done": True, "exit_code": -1}
            return

    # Defense-in-depth: sanitize even if a caller passes a dict that started
    # from os.environ — sandbox_env() only keeps the safe allowlist regardless.
    proc = await asyncio.create_subprocess_shell(
        command, cwd=cwd, env=sandbox_env(env),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    async for item in _stream_proc(proc, timeout):
        yield item


async def run_python(code: str, cwd: str, timeout: int = 300) -> str:
    """
    Execute a Python script string in a subprocess.
    Returns combined stdout + stderr, truncated to 8000 chars.
    """
    Path(cwd).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=cwd, delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name
    try:
        proc = await asyncio.create_subprocess_exec(
            *_guarded_launch_args(sys.executable, tmp),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=sandbox_env(),
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"TIMEOUT: Script exceeded {timeout}s."
        return ((out.decode("utf-8", "replace") + err.decode("utf-8", "replace")).strip() or "(no output)")[:8000]
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


async def auto_install_and_retry(code: str, error: str, cwd: str) -> str | None:
    """
    If error is a ModuleNotFoundError, install the missing package and retry.
    Returns new output if install+retry succeeded, None if not a package error.
    """
    if "No module named" not in error:
        return None
    try:
        pkg_raw = error.split("No module named '")[1].split("'")[0].split(".")[0]
        pkg_map = {"fpdf": "fpdf2", "docx": "python-docx", "pptx": "python-pptx"}
        pkg = pkg_map.get(pkg_raw, pkg_raw)
    except IndexError:
        return None
    
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", pkg, "--quiet",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sandbox_env(),
        )
        await asyncio.wait_for(proc.communicate(), timeout=120)
        return await run_python(code, cwd)
    except Exception:
        return None


def extract_python_blocks(text: str) -> list[str]:
    """Extract all ```python ... ``` code blocks from text."""
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


async def run_shell(cmd: str, cwd: str, blocked_patterns: list[str] | None = None,
                     env: dict | None = None) -> str:
    """
    Execute a shell command in a subprocess.
    blocked_patterns: list of substrings that block the command entirely.
    """
    if blocked_patterns:
        cl = cmd.lower()
        if any(b in cl for b in blocked_patterns):
            return "BLOCKED: Command contains a forbidden pattern."
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sandbox_env(env),
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=30)
        return ((out.decode("utf-8", "replace") + err.decode("utf-8", "replace")).strip() or "(no output)")[:8000]
    except asyncio.TimeoutError:
        return "TIMEOUT: Command exceeded 30s."
    except Exception as e:
        return f"ERROR: {e}"
