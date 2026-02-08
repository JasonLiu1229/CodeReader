from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .prompts import DEFAULT_GRADE_PROMPT, DEFAULT_HEALTH_PROMPT
from .runner import clamp_score, extract_json_object, GradeResult, RunnerResult


def _post_json(
    *,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_s: float,
) -> tuple[int, str, Optional[Dict[str, Any]]]:
    """
    Blocking HTTP helper (runs in a thread).
    Returns: (status_code, response_text, parsed_json_or_none)
    """
    r = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    text = r.text or ""
    try:
        data = r.json()
    except Exception:
        data = None
    return r.status_code, text, data


def _extract_chat_content(data: Dict[str, Any]) -> Optional[str]:
    """
    Supports the classic chat.completions response shape:
      { choices: [ { message: { content: "..." } } ] }
    """
    choices = data.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    return content if isinstance(content, str) else None


def _extract_error_message(data: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Best-effort extraction of OpenAI-style error payloads:
      { error: { message: "..." } }
    """
    if not isinstance(data, dict):
        return None
    err = data.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        return msg if isinstance(msg, str) and msg.strip() else None
    return None


@dataclass(frozen=True)
class OpenAIRunner:
    name: str
    api_key: str
    model: Optional[str] = None

    grade_prompt_template: str = DEFAULT_GRADE_PROMPT
    health_prompt: str = DEFAULT_HEALTH_PROMPT

    base_url: str = "https://api.openai.com/v1"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _model_name(self) -> str:
        return self.model or "gpt-4.1-mini"

    def _chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    async def health_check(self, timeout_s: float = 15.0) -> RunnerResult:
        loop = asyncio.get_running_loop()
        start = loop.time()

        payload: Dict[str, Any] = {
            "model": self._model_name(),
            "messages": [{"role": "user", "content": self.health_prompt}],
        }

        try:
            status, text, data = await asyncio.to_thread(
                _post_json,
                url=self._chat_completions_url(),
                payload=payload,
                headers=self._headers(),
                timeout_s=timeout_s,
            )
        except requests.Timeout:
            return RunnerResult(
                ok=False,
                stdout="",
                stderr="",
                exit_code=124,
                duration_s=loop.time() - start,
                error="API request timed out",
            )
        except requests.RequestException as e:
            return RunnerResult(
                ok=False,
                stdout="",
                stderr="",
                exit_code=1,
                duration_s=loop.time() - start,
                error=str(e),
            )

        dur = loop.time() - start

        if status < 200 or status >= 300:
            return RunnerResult(
                ok=False,
                stdout="",
                stderr=text,
                exit_code=status,
                duration_s=dur,
                error=_extract_error_message(data) or f"HTTP {status}",
            )

        if not isinstance(data, dict):
            return RunnerResult(
                ok=False,
                stdout="",
                stderr=text,
                exit_code=1,
                duration_s=dur,
                error="Invalid JSON response (not an object)",
            )

        content = _extract_chat_content(data)
        if content is None:
            return RunnerResult(
                ok=False,
                stdout="",
                stderr=text,
                exit_code=1,
                duration_s=dur,
                error="Healthcheck failed: could not extract assistant content",
            )

        if content.strip() != "OK":
            return RunnerResult(
                ok=False,
                stdout=content,
                stderr=text,
                exit_code=1,
                duration_s=dur,
                error="Healthcheck failed: reply was not exactly 'OK'",
            )

        return RunnerResult(
            ok=True,
            stdout=content,
            stderr="",
            exit_code=0,
            duration_s=dur,
            error=None,
        )

    async def grade_code(
        self,
        code: str,
        *,
        tags: List[str],
        language: str,
        rules: List[str] = None,
        timeout_s: float = 120.0,
        max_output_tokens: Optional[int] = None,
    ) -> GradeResult:
        tags_str = ", ".join(tags)

        if not rules:
            rules_str = "No specific rules specified"
        else:
            rules_str = ", ".join(rules)

        prompt = self.grade_prompt_template.format(
            tags=tags_str, language=language, code=code, rules=rules_str
        )

        payload: Dict[str, Any] = {
            "model": self._model_name(),
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = int(max_output_tokens)

        try:
            status, text, data = await asyncio.to_thread(
                _post_json,
                url=self._chat_completions_url(),
                payload=payload,
                headers=self._headers(),
                timeout_s=timeout_s,
            )
        except requests.Timeout:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout="",
                raw_stderr="",
                parsed=None,
                rationale="",
                error=f"Timeout after {timeout_s:.1f}s",
            )
        except requests.RequestException as e:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout="",
                raw_stderr="",
                parsed=None,
                rationale="",
                error=str(e),
            )

        if status < 200 or status >= 300:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout="",
                raw_stderr=text,
                parsed=data if isinstance(data, dict) else None,
                rationale="",
                error=_extract_error_message(data) or f"HTTP {status}",
            )

        if not isinstance(data, dict):
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout="",
                raw_stderr=text,
                parsed=None,
                rationale="",
                error="Invalid JSON response (not an object)",
            )

        content = _extract_chat_content(data)
        if content is None:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout="",
                raw_stderr=text,
                parsed=data,
                rationale="",
                error="Could not extract assistant content",
            )

        parsed = extract_json_object(content)
        if not parsed:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=content,
                raw_stderr=text,
                parsed=None,
                rationale="",
                error="Could not find/parse JSON object in model output",
            )

        score = clamp_score(parsed.get("score"))
        rationale = parsed.get("rationale")

        if score is None:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=content,
                raw_stderr=text,
                parsed=parsed,
                rationale=str(rationale or ""),
                error="JSON parsed but 'score' missing or not numeric",
            )

        return GradeResult(
            model_name=self.name,
            score=score,
            raw_stdout=content,
            raw_stderr=text,
            parsed=parsed,
            rationale=str(rationale or ""),
            error=None,
        )
