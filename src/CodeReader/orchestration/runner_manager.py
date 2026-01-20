from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from codereader.runners.ollama_runner import OllamaRunner
from codereader.runners.runner import BaseRunner, RunnerResult


@dataclass(frozen=True)
class Settings:
    timeout_seconds: float = 120.0
    health_timeout_seconds: float = 15.0
    max_concurrency: int = 2
    fail_on_no_active_models: bool = True
    log_path: str = "readability_log.txt"
    debug_jsonl_path: str = "responses.jsonl"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    runner: str
    runner_config: Dict[str, Any]
    weight: float = 1.0


@dataclass(frozen=True)
class AppConfig:
    language: str
    tags: List[str]
    settings: Settings
    models: List[ModelConfig]


@dataclass(frozen=True)
class HealthReportItem:
    model_name: str
    runner: str
    ok: bool
    error: Optional[str]
    exit_code: int
    stderr: str
    stdout: str
    duration_s: float


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required field: '{key}'")
    return d[key]


def parse_config_dict(raw: Dict[str, Any]) -> AppConfig:
    language = str(_require(raw, "language"))
    tags = list(_require(raw, "tags"))

    settings_raw = raw.get("settings", {}) or {}
    settings = Settings(
        timeout_seconds=float(settings_raw.get("timeout_seconds", 120)),
        health_timeout_seconds=float(settings_raw.get("health_timeout_seconds", 15)),
        max_concurrency=int(settings_raw.get("max_concurrency", 2)),
        fail_on_no_active_models=bool(
            settings_raw.get("fail_on_no_active_models", True)
        ),
        log_path=str(settings_raw.get("log_path", "readability_log.txt")),
        debug_jsonl_path=str(settings_raw.get("debug_jsonl_path", "responses.jsonl")),
    )
    
    if settings.max_concurrency < 1:
        raise ValueError("settings.max_concurrency must be >= 1")
    
    models_raw = list(_require(raw, "models"))
    models: List[ModelConfig] = []

    for i, model in enumerate(models_raw):
        if not isinstance(model, dict):
            raise ValueError(f"models[{i}] must be a mapping/object")
        
        runner_config = model.get("runner_config", {}) or {}
        
        models.append(
            ModelConfig(
                name=str(_require(model, "name")),
                model=str(_require(model, "model")),
                runner=str(_require(model, "runner")),
                runner_config=dict(runner_config),
                weight=float(model.get("weight", 1.0)),
            )
        ) 
    
    return AppConfig(language=language, tags=tags, settings=settings, models=models)

def load_config_yaml(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return parse_config_dict(raw)

class RunnerManager:
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._all_runners: List[BaseRunner] = []
        self.active_runners: List[BaseRunner] = []
        self.report: List[HealthReportItem] = []

    def build_runners(self) -> List[BaseRunner]:
        runners: List[BaseRunner] = []
        
        for m in self.config.models:
            runner_type = m.runner.lower().strip()

            if runner_type == "ollama":
                ollama_bin = str(m.runner_config.get("ollama_bin", "ollama"))
                runners.append(
                    OllamaRunner(
                        name=m.name,
                        model=m.model,
                        ollama_bin=ollama_bin,
                    )
                )
            else:
                # TODO: add "llamacpp" / "transformers" here
                raise ValueError(f"Unsupported runner type: '{m.runner}' for model '{m.name}'")
        
        self._all_runners = runners
        return runners
        
    async def initialize(self) -> None:
        self.build_runners()

        sem = asyncio.Semaphore(self.config.settings.max_concurrency)

        async def check_one(runner: BaseRunner) -> Tuple[BaseRunner, RunnerResult]:
            async with sem:
                res = await runner.health_check(timeout_s=self.config.settings.health_timeout_seconds)
                return runner, res

        tasks = [asyncio.create_task(check_one(r)) for r in self._all_runners]
        results = await asyncio.gather(*tasks)
        
        active: List[BaseRunner] = []
        report: List[HealthReportItem] = []
        
        for runner, res in results:
            report.append(
                HealthReportItem(
                    model_name=getattr(runner, "name", "<unknown>"),
                    runner=runner.__class__.__name__,
                    ok=res.ok,
                    error=res.error,
                    exit_code=res.exit_code,
                    stderr=res.stderr,
                    stdout=res.stdout,
                    duration_s=res.duration_s,
                )
            )
            if res.ok:
                active.append(runner)
        
        self.active_runners = active
        self.report = report
        
        if self.config.settings.fail_on_no_active_models and not self.active_runners:
            errors = "\n".join(
                f"- {r.model_name} ({r.runner}): {r.error or 'healthcheck failed'}"
                for r in self.report
            )
            raise RuntimeError(f"No active models after health checks.\n{errors}")
        
    def get_context(self) -> Tuple[str, List[str]]:
        return self.config.language, self.config.tags
