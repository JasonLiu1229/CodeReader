from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .prompts import DEFAULT_GRADE_PROMPT, DEFAULT_HEALTH_PROMPT

from .runner import (
    clamp_score,
    extract_json_object,
    GradeResult,
    run_subprocess,
    RunnerResult,
)
from .utils import format_prompt, compute_weighted_score

_SERVER_CHECK: RunnerResult | None = None


def _looks_like_ollama_server_down(
    stderr: str, stdout: str, error: str | None = None
) -> bool:
    text = f"{stdout}\n{stderr}\n{error or ''}".lower()
    patterns = [
        "could not connect",
        "connection refused",
        "connect: connection refused",
        "dial tcp",
        "no such host",
        "connectex",
        "actively refused",
        "ollama server not responding",
        "could not locate ollama app",
        "failed to connect",
        "error: eof",
        "timed out waiting for server to start",
        "timeout waiting for server",
        "timeout after",
    ]
    return any(p in text for p in patterns)


@dataclass(frozen=True)
class OllamaRunner:
    name: str  # just as an username (for like eas of use)
    model: str  # ollama model identifier
    ollama_bin: str = "ollama"
    grade_prompt_template: str = DEFAULT_GRADE_PROMPT
    health_prompt: str = DEFAULT_HEALTH_PROMPT
    no_think: bool = False

    async def _check_server_running(self, timeout_s: float = 3.0) -> RunnerResult:
        """
        Fast preflight check.
        """
        global _SERVER_CHECK

        if _SERVER_CHECK is not None and _SERVER_CHECK.ok:
            return _SERVER_CHECK

        cmd = [self.ollama_bin, "list"]
        res = await run_subprocess(cmd, timeout_s=timeout_s)

        if res.ok:
            _SERVER_CHECK = res
            return res

        if _looks_like_ollama_server_down(res.stderr, res.stdout, res.error):
            return RunnerResult(
                ok=False,
                stdout=res.stdout,
                stderr=res.stderr,
                exit_code=res.exit_code,
                duration_s=res.duration_s,
                error=(
                    "Ollama server doesn't seem to be running.\n"
                    "Start it first with: `ollama serve`\n"
                    "Then retry your command."
                ),
            )

        _SERVER_CHECK = res
        return res

    async def health_check(self, timeout_s: float = 15.0) -> RunnerResult:
        pre = await self._check_server_running(timeout_s=min(3.0, timeout_s))
        if not pre.ok:
            return pre

        cmd = [self.ollama_bin, "run", self.model, self.health_prompt]
        res = await run_subprocess(cmd, timeout_s=timeout_s)

        if not res.ok:
            return res

        if "OK" not in res.stdout:
            return RunnerResult(
                ok=False,
                stdout=res.stdout,
                stderr=res.stderr,
                exit_code=res.exit_code,
                duration_s=res.duration_s,
                error="Healthcheck failed: 'OK' not found in output",
            )

        return res

    async def grade_code(
        self,
        code: str,
        *,
        tags: List[str],
        language: str,
        rules: Optional[List[str]] = None,
        timeout_s: float = 120.0,
        max_output_tokens: Optional[int] = None,
    ) -> GradeResult:
        prompt = format_prompt(self.grade_prompt_template, tags, rules, language, code)

        cmd = (
            [self.ollama_bin, "run"]
            + (["--think=false"] if self.no_think else [])
            + [self.model]
        )

        res = await run_subprocess(cmd, stdin_text=prompt, timeout_s=timeout_s)
        print(f"[DEBUG {self.name}] raw output: {repr(res.stdout[:800])}", flush=True)
        if not res.ok:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=res.stdout,
                raw_stderr=res.stderr,
                parsed=None,
                error=res.error or f"ollama exited with {res.exit_code}",
                rationale="",
            )

        parsed = extract_json_object(res.stdout)
        if not parsed:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=res.stdout,
                raw_stderr=res.stderr,
                parsed=None,
                error="Could not find/parse JSON object in model output",
                rationale="",
            )

        score = compute_weighted_score(parsed) or clamp_score(parsed.get("score"))
        rationale = parsed.get("rationale", "")
        if score is None:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=res.stdout,
                raw_stderr=res.stderr,
                parsed=parsed,
                error="JSON parsed but 'score' missing or not numeric",
                rationale=rationale,
            )

        return GradeResult(
            model_name=self.name,
            score=score,
            raw_stdout=res.stdout,
            raw_stderr=res.stderr,
            parsed=parsed,
            error=None,
            rationale=rationale,
        )
