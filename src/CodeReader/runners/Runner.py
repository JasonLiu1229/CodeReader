from __future__ import annotations

import asyncio
import json
import re
import os

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import requests


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


class BaseRunner(
    Protocol
):  # So if we implement other runners they need to atleast adhere to these things (contracting)
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
        rules: Optional[List[str]] = None,
        timeout_s: float = 120.0,
        max_output_tokens: Optional[int] = None,
    ) -> GradeResult: ...


def _post_request(
    *,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_s: float,
) -> requests.Response:
    return requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout_s,
    )


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

    clean_env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", **(env or {})}

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_text is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=clean_env,
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


def _try_parse_from(text: str, start: int) -> Optional[Dict[str, Any]]:
    """Attempt to parse a JSON object starting at `start` using brace-depth scan."""
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None


def clean_output(text: str):
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    text = re.sub(r"Thinking\.\.\..*?\.\.\.done thinking\.", "", text, flags=re.DOTALL)

    return text


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract the first valid JSON object from model output.

    Handles:
    - <think>...</think> XML blocks (DeepSeek-R1 style)
    - Plain "Thinking..." preamble blocks (Qwen3 via ollama).
      Qwen3 thinking text may contain { characters (e.g. Java code snippets),
      so we try ALL candidate { positions and return the first valid parse
      that contains the expected keys rather than stopping at the first {.
    - Markdown fences: ```json { ... } ``` (DeepSeek, Phi)
    - Trailing explanation text after the JSON (DeepSeek)
    - Nested objects: {"subscores": {"a": 1}}
    """

    text = clean_output(text)

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        blob = fenced.group(1).strip()
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    candidates = [i for i, ch in enumerate(text) if ch == "{"]
    for start in candidates:
        parsed = _try_parse_from(text, start)
        if parsed is None:
            continue

        if any(k in parsed for k in ("score", "subscores", "scores", "rationale")):
            return parsed

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
