from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .runner import (
    clamp_score,
    GradeResult,
    run_subprocess,
    RunnerResult,
    extract_json_object
)

from .prompts import DEFAULT_GRADE_PROMPT, DEFAULT_HEALTH_PROMPT


@dataclass(frozen=True)
class OllamaRunner:
    name: str  # just as an username (for like eas of use)
    model: str  # ollama model identifier
    ollama_bin: str = "ollama"
    grade_prompt_template: str = DEFAULT_GRADE_PROMPT
    health_prompt: str = DEFAULT_HEALTH_PROMPT

    async def health_check(self, timeout_s: float = 15.0) -> RunnerResult: 
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
        timeout_s: float = 120.0,
        max_output_tokens: Optional[int] = None,
    ) -> GradeResult:
        tags_str = ", ".join(tags)
        
        prompt = self.grade_prompt_template.format(tags=tags_str, language=language, code=code)

        cmd = [self.ollama_bin, "run", self.model]
        
        res = await run_subprocess(cmd, stdin_text=prompt, timeout_s=timeout_s)
        
        if not res.ok:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=res.stdout,
                raw_stderr=res.stderr,
                parsed=None,
                error=res.error or f"ollama exited with {res.exit_code}",
            )
        
        parsed = extract_json_object(res.stdout)
