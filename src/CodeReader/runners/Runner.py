from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, List


@dataclass(frozen=True)
class RunnerResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float
    error: Optional[str] = None


@dataclass(frozen=True)
class GradeResult:
    model_name: str
    score: Optional[int]  # None if failed
    raw_stdout: str
    raw_stderr: str
    parsed: Optional[Dict[str, Any]]
    rationale: str
    error: Optional[str] = None


class BaseRunner(Protocol): # So if we implement other runners they need to atleast adhere to these things (contracting)
    """
    Minimal runner contract.
    Each runner knows how to:
      - health check a model
      - grade code (return a numeric score 0..100)
    """

    name: str

    async def health_check(self, timeout_s: float = 15.0) -> RunnerResult: ...

    async def grade_code(
        self,
        code: str,
        *,
        tags: List[str],
        language: str,
        timeout_s: float = 120.0,
        max_output_tokens: Optional[int] = None,
    ) -> GradeResult: ...


async def run_subprocess(
    cmd: list[str],
    *,
    stdin_text: Optional[str] = None,
    timeout_s: float = 60.0,
    env: Optional[Dict[str, str]] = None,
) -> RunnerResult:
    """
    Async subprocess runner:
    - writes stdin
    - captures stdout/stderr
    - enforces timeout
    - kills process on timeout
    """
    loop = asyncio.get_running_loop()
    start = loop.time()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_text is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError as e:
        dur = loop.time() - start
        return RunnerResult(
            ok=False,
            stdout="",
            stderr="",
            exit_code=127,
            duration_s=dur,
            error=f"Executable not found: {cmd[0]} ({e})",
        )
    except Exception as e:
        dur = loop.time() - start
        return RunnerResult(
            ok=False,
            stdout="",
            stderr="",
            exit_code=1,
            duration_s=dur,
            error=f"Failed to start subprocess: {e}",
        )

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(
                input=stdin_text.encode("utf-8") if stdin_text is not None else None
            ),
            timeout=timeout_s,
        )
        exit_code = proc.returncode or 0
        dur = loop.time() - start
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        ok = exit_code == 0
        return RunnerResult(
            ok=ok, stdout=stdout, stderr=stderr, exit_code=exit_code, duration_s=dur
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass

        try:
            await proc.wait()
        except Exception:
            pass

        dur = loop.time() - start
        return RunnerResult(
            ok=False,
            stdout="",
            stderr="",
            exit_code=124,
            duration_s=dur,
            error=f"Timeout after {timeout_s:.1f}s",
        )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    blob = m.group(0).strip()
    try:
        return json.loads(blob)
    except Exception:
        return None


def clamp_score(score: Any) -> Optional[int]:
    try:
        if isinstance(score, bool):
            return None
        if isinstance(score, (int, float)):
            v = int(round(float(score)))
        elif isinstance(score, str):
            v = int(round(float(score.strip())))
        else:
            return None
        return max(0, min(100, v))
    except Exception:
        return None
