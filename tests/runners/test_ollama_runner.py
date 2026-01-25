import codereader.runners.ollama_runner as ollama_mod
import pytest

from codereader.runners.ollama_runner import OllamaRunner


@pytest.mark.anyio
async def test_health_check_success_contains_ok(monkeypatch):
    async def fake_run_subprocess(
        cmd, *, stdin_text=None, timeout_s=60.0, env=None
    ) -> ollama_mod.RunnerResult:
        return ollama_mod.RunnerResult(
            ok=True,
            stdout="OK\n",
            stderr="",
            exit_code=0,
            duration_s=0.01,
            error=None,
        )

    monkeypatch.setattr(ollama_mod, "run_subprocess", fake_run_subprocess)

    runner = OllamaRunner(name="llama3", model="llama3")
    res = await runner.health_check(timeout_s=1.0)

    assert res.ok is True
    assert res.error is None


@pytest.mark.anyio
async def test_health_check_propagates_subprocess_failure(monkeypatch):
    async def fake_run_subprocess(
        cmd, *, stdin_text=None, timeout_s=60.0, env=None
    ) -> ollama_mod.RunnerResult:
        return ollama_mod.RunnerResult(
            ok=False,
            stdout="",
            stderr="something went wrong",
            exit_code=127,
            duration_s=0.01,
            error="Executable not found: ollama",
        )

    monkeypatch.setattr(ollama_mod, "run_subprocess", fake_run_subprocess)

    runner = OllamaRunner(name="llama3", model="llama3", ollama_bin="ollama")
    res = await runner.health_check(timeout_s=1.0)

    assert res.ok is False
    assert res.exit_code == 127
    assert "Executable not found" in (res.error or "")
