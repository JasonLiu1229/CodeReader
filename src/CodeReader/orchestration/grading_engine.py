from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from codereader.orchestration.runner_manager import AppConfig

from codereader.runners.runner import BaseRunner, GradeResult


@dataclass(frozen=True)
class ModelGrade:
    model_name: str
    score: Optional[int]
    rationale: str
    weight: float
    error: Optional[str] = None
    parsed: Optional[Dict] = None


@dataclass(frozen=True)
class FileGrade:
    filename: str
    language: str
    tags: List[str]
    grades: List[ModelGrade]
    average: Optional[float]
    weighted_average: Optional[float]


class GradingEngine:
    """
    Runs readability grading across active runners.
    """

    def __init__(self, *, config: AppConfig, active_runners: Sequence[BaseRunner]):
        self.config = config
        self.runners: List[BaseRunner] = list(active_runners)
        self.weights: Dict[str, float] = {}
        for m in self.config.models:
            self.weights[m.name] = float(getattr(m, "weight", 1.0) or 1.0)

    def _weight_for(self, runner: BaseRunner) -> float:
        return float(self.weights.get(getattr(runner, "name", ""), 1.0))

    async def grade_one(
        self,
        *,
        filename: str,
        code: str,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
    ) -> FileGrade:
        """
        Grade a single code blob.
        """
        if tags is None:
            tags = list(self.config.tags)

        if language is None:
            language = self.config.language

        sem = asyncio.Semaphore(self.config.settings.max_concurrency)

        async def run_one(runner: BaseRunner) -> Tuple[BaseRunner, GradeResult]:
            async with sem:
                res = await runner.grade_code(
                    code,
                    tags=tags,
                    language=language,
                    timeout_s=self.config.settings.timeout_seconds,
                )
                return runner, res

        tasks = [asyncio.create_task(run_one(r)) for r in self.runners]
        results = await asyncio.gather(*tasks)

        model_grades: List[ModelGrade] = []
        successful_scores: List[int] = []
        weighted_sum = 0.0
        weight_total = 0.0

        for runner, res in results:
            w = self._weight_for(runner)
            score = res.score
            rationale = res.rationale

            model_grades.append(
                ModelGrade(
                    model_name=res.model_name,
                    score=score,
                    weight=w,
                    error=res.error,
                    parsed=res.parsed,
                    rationale=rationale
                )
            )

            if score is not None:
                successful_scores.append(score)
                weighted_sum += score * w
                weight_total += w

        avg = (
            (sum(successful_scores) / len(successful_scores))
            if successful_scores
            else None
        )
        wavg = (weighted_sum / weight_total) if weight_total > 0 else None

        return FileGrade(
            filename=filename,
            language=language,
            tags=tags,
            grades=model_grades,
            average=avg,
            weighted_average=wavg,
        )

    async def grade_many(
        self,
        *,
        items: Sequence[Tuple[str, str]],
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
    ) -> List[FileGrade]:
        """
        Grade multiple (filename, code) pairs.
        """
        out: List[FileGrade] = []
        for filename, code in items:
            out.append(
                await self.grade_one(
                    filename=filename, code=code, tags=tags, language=language
                )
            )
        return out

    @staticmethod
    def grades_array(file_grade: FileGrade) -> List[int]:
        """
        Returns a simple list of successful integer scores, in model order.
        """
        return [g.score for g in file_grade.grades if g.score is not None]

    @staticmethod
    def format_log_line(file_grade: FileGrade) -> str:
        """
        Formats based on:
          filename, [grades], avg([grades]) = total
        """
        grades = GradingEngine.grades_array(file_grade)
        if grades:
            avg = sum(grades) / len(grades)
            return f"{file_grade.filename}, {grades}, avg({grades}) = {avg:.2f}"
        return f"{file_grade.filename}, [], avg([]) = N/A"
