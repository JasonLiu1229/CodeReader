from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .prompts import DEFAULT_GRADE_PROMPT, DEFAULT_HEALTH_PROMPT

from .runner import (
    clamp_score,
    extract_json_object,
    GradeResult,
    run_apicall,
    RunnerResult,
)


def _normalize_endpoint(api_url: str) -> str:
    base = api_url.rstrip("/")
    if base.endswith("/v1/chat/completions") or base.endswith("/chat/completions"):
        return base
    return f"{base}/v1/chat/completions"


def _extract_choice_content(resp_json: Dict[str, Any]) -> Optional[str]:
    try:
        choices = resp_json.get("choices") or []
        if not choices:
            return None
        ch0 = choices[0] or {}

        msg = ch0.get("message")
        if isinstance(msg, dict) and "content" in msg:
            return msg.get("content")

        if "text" in ch0:
            return ch0.get("text")

        return None
    except Exception:
        return None


@dataclass(frozen=True)
class ApiRunner:
    name: str
    api_url: str
    api_key: str
    model: Optional[str] = None
    grade_prompt_template: str = DEFAULT_GRADE_PROMPT
    health_prompt: str = DEFAULT_HEALTH_PROMPT

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return headers

    def _model_id(self) -> str:
        return self.model or self.name

    async def health_check(self, timeout_s: float = 15.0) -> RunnerResult:
        url = _normalize_endpoint(self.api_url)

        payload = {
            "model": self._model_id(),
            "messages": [{"role": "user", "content": self.health_prompt}],
            "temperature": 0.0,
            "max_tokens": 16,
        }

        res = await run_apicall(
            url=url,
            payload=payload,
            headers=self._headers(),
            timeout_s=timeout_s,
        )

        if not res.ok:
            return res

        try:
            data = json.loads(res.stdout)
            content = _extract_choice_content(data) or ""
            if "OK" not in content:
                return RunnerResult(
                    ok=False,
                    stdout=res.stdout,
                    stderr=res.stderr,
                    exit_code=1,
                    duration_s=res.duration_s,
                    error="Healthcheck failed: 'OK' not found in output",
                )
        except Exception:
            return RunnerResult(
                ok=False,
                stdout=res.stdout,
                stderr=res.stderr,
                exit_code=1,
                duration_s=res.duration_s,
                error="Healthcheck failed: could not parse API response as expected",
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

        prompt = self.grade_prompt_template.format(
            tags=tags_str, language=language, code=code
        )

        url = _normalize_endpoint(self.api_url)

        payload: Dict[str, Any] = {
            "model": self._model_id(),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict code readability evaluator, output based on the givem prompt.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens

        res = await run_apicall(
            url=url,
            payload=payload,
            headers=self._headers(),
            timeout_s=timeout_s,
        )

        if not res.ok:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=res.stdout,
                raw_stderr=res.stderr,
                parsed=None,
                error=res.error or f"api call failed (exit_code={res.exit_code})",
                rationale=None,
            )

        try:
            resp_json = json.loads(res.stdout)
        except Exception:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=res.stdout,
                raw_stderr=res.stderr,
                parsed=None,
                error="API returned non-JSON response",
                rationale=None,
            )

        content = _extract_choice_content(resp_json)
        if not content:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=res.stdout,
                raw_stderr=res.stderr,
                parsed=None,
                error="API response missing model text (choices[0])",
                rationale=None,
            )

        parsed = extract_json_object(content) or (
            json.loads(content) if content.strip().startswith("{") else None
        )
        if not parsed:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=content,
                raw_stderr=res.stderr,
                parsed=None,
                error="Could not find/parse JSON object in model output",
                rationale=None,
            )

        score = clamp_score(parsed.get("score"))
        rationale = parsed.get("rationale")

        if score is None:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=content,
                raw_stderr=res.stderr,
                parsed=parsed,
                error="JSON parsed but 'score' missing or not numeric",
                rationale=rationale if isinstance(rationale, str) else None,
            )

        return GradeResult(
            model_name=self.name,
            score=score,
            raw_stdout=content,
            raw_stderr=res.stderr,
            parsed=parsed,
            error=None,
            rationale=rationale if isinstance(rationale, str) else None,
        )
