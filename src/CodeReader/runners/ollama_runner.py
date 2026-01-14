from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base_runner import (
    clamp_score,
    extract_first_json_object,
    GradeResult,
    run_subprocess,
    RunnerResult,
)

from .prompts import DEFAULT_GRADE_PROMPT, DEFAULT_HEALTH_PROMPT


@dataclass(frozen=True)
class OllamaRunner:
    name: str  # just as an username (for like eas of use)
    model: str  # ollama model identifier
    ollama_bin: str = "ollama"
    grade_prompt_template: str = DEFAULT_GRADE_PROMPT
    health_prompt: str = DEFAULT_HEALTH_PROMPT

    async def health_check(self, timeout_s: float = 15.0) -> RunnerResult: ...

    async def grade_code(
        self,
        code: str,
        *,
        tag: str,
        language: str,
        timeout_s: float = 120.0,
        max_output_tokens: Optional[int] = None,
    ) -> GradeResult: ...
