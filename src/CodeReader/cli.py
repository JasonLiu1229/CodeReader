from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.table import Table

from codereader.logger.result_logger import ResultLogger

from codereader.orchestration.grading_engine import FileGrade, GradingEngine
from codereader.orchestration.runner_manager import load_config_yaml, RunnerManager

app = typer.Typer(add_completion=False, help="CodeReader")


@app.callback()
def main() -> None:
    """CodeReader command line interface."""
    pass


def _read_code(
    file: Optional[Path], text: Optional[str], stdin: bool
) -> tuple[str, str]:
    """
    Returns (display_name, code).
    Priority: file > text > stdin.
    """
    if file is not None:
        code = file.read_text(encoding="utf-8")
        return str(file), code

    if text is not None:
        return "inline", text

    if stdin:
        code = typer.get_text_stream("stdin").read()
        return "stdin", code

    raise typer.BadParameter("Provide one of: --file, --text, or --stdin")


def _print_result(result: FileGrade) -> None:
    table = Table(title=f"Readability grades: {result.filename}", show_lines=True)
    table.add_column("Model")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Rationale", justify="right")
    table.add_column("Error")

    for g in result.grades:
        table.add_row(
            g.model_name,
            "-" if g.score is None else str(g.score),
            f"{g.weight:g}",
            "-" if g.rationale is None else g.rationale,
            "" if not g.error else g.error,
        )

    print(table)
    print(
        f"[bold]Average:[/bold] {result.average if result.average is not None else 'N/A'}"
    )
    print(
        f"[bold]Weighted average:[/bold] {result.weighted_average if result.weighted_average is not None else 'N/A'}"
    )


@app.command("grade")
def grade(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, dir_okay=False, readable=True
    ),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", exists=True, dir_okay=False, readable=True
    ),
    text: Optional[str] = typer.Option(None, "--text", help="Inline code to grade."),
    stdin: bool = typer.Option(False, "--stdin", help="Read code from stdin."),
    name: Optional[str] = typer.Option(
        None, "--name", help="Label used as 'filename' in the log."
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Disable console output"),
) -> None:
    """
    Grade a single code input using the YAML config.
    """
    display_name, code = _read_code(file, text, stdin)
    filename = name or display_name

    cfg = load_config_yaml(str(config))

    async def _run() -> FileGrade:
        manager = RunnerManager(cfg)
        await manager.initialize()

        engine = GradingEngine(config=cfg, active_runners=manager.active_runners)
        return await engine.grade_one(
            filename=filename,
            code=code,
            tags=cfg.tags,
            language=cfg.language,
        )

    result = asyncio.run(_run())

    logger = ResultLogger(log_path=cfg.settings.log_path)
    logger.append_result(result)

    if not quiet:
        _print_result(result)
        print(f"[green]Logged:[/green] {cfg.settings.log_path}")


if __name__ == "__main__":
    app()
