from __future__ import annotations
from pathlib import Path

from codereader.orchestration.grading_engine import FileGrade, GradingEngine


class ResultLogger:
    def __init__(self, *, log_path: str):
        self.log_path = Path(log_path)

    def append_result(self, result: FileGrade) -> None:
        line = GradingEngine.format_log_line(result)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
