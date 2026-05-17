from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .prompts import DEFAULT_GRADE_PROMPT, DEFAULT_HEALTH_PROMPT
from .runner import (
    GradeResult,
    RunnerResult,
    clamp_score,
    extract_json_object,
)
from .utils import compute_weighted_score, format_prompt


def _import_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        return torch, AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as e:
        raise ImportError(
            "The 'transformers' extra is required for TransformersRunner.\n"
            "Install it with: pip install codereader[transformers]\n"
            f"Original error: {e}"
        ) from e


def _import_peft():
    try:
        from peft import PeftModel

        return PeftModel
    except ImportError as e:
        raise ImportError(
            "The 'peft' package is required to load adapter weights.\n"
            "Install it with: pip install peft\n"
            f"Original error: {e}"
        ) from e


@dataclass
class TransformersRunner:
    """
    A runner that loads a HuggingFace causal-LM model (optionally with a
    LoRA / PEFT adapter) and uses it to grade code readability.

    Config example
    --------------
    runners:
      - type: transformers
        name: my-finetuned-codebert
        model: mistralai/Mistral-7B-Instruct-v0.2   # HF model id or local path
        adapter_path: ./checkpoints/lora-epoch3      # optional LoRA adapter dir
        device: cuda                                 # cuda | cpu | mps | auto
        torch_dtype: float16                         # float32 | float16 | bfloat16
        max_new_tokens: 512
        temperature: 0.1
        no_think: false
    """

    name: str  # display name / identifier
    model: str  # HF model id or local path
    adapter_path: Optional[str] = None  # optional: path to PEFT/LoRA adapter dir
    device: str = "auto"  # "cuda", "cpu", "mps", or "auto"
    torch_dtype: str = "float16"  # "float32", "float16", "bfloat16"
    max_new_tokens: int = 512
    temperature: float = 0.1
    do_sample: bool = True
    grade_prompt_template: str = DEFAULT_GRADE_PROMPT
    health_prompt: str = DEFAULT_HEALTH_PROMPT
    no_think: bool = False

    _pipeline: object = field(default=None, init=False, repr=False, compare=False)

    def _dtype(self):
        import torch

        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.torch_dtype, torch.float16)

    def _resolve_device(self):
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _build_pipeline(self):
        """Lazily build (and cache) the HuggingFace text-generation pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        _, AutoModelForCausalLM, AutoTokenizer, pipeline = _import_transformers()
        device = self._resolve_device()
        dtype = self._dtype()

        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=dtype,
            device_map=device if device == "auto" else None,
            trust_remote_code=True,
        )

        if self.adapter_path:
            PeftModel = _import_peft()
            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.merge_and_unload()

        if device not in ("auto",):
            model = model.to(device)

        model.eval()

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=None if device == "auto" else (0 if device == "cuda" else device),
        )

        self._pipeline = pipe
        return pipe

    def _generate(self, prompt: str) -> str:
        """Run a single forward pass and return the *new* tokens as a string."""
        pipe = self._build_pipeline()

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else 1.0,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False,
        )

        outputs = pipe(prompt, **gen_kwargs)

        return outputs[0]["generated_text"] if outputs else ""

    async def health_check(self, timeout_s: float = 60.0) -> RunnerResult:
        """
        Verify the model loads and can produce a coherent output.
        We do this synchronously because HF inference is blocking anyway –
        wrap it so the async interface is satisfied.
        """
        start = time.monotonic()
        try:
            output = self._generate(self.health_prompt)
            duration = time.monotonic() - start
            if "OK" not in output:
                return RunnerResult(
                    ok=False,
                    stdout=output,
                    stderr="",
                    exit_code=1,
                    duration_s=duration,
                    error="Health check failed: 'OK' not found in model output",
                )
            return RunnerResult(
                ok=True,
                stdout=output,
                stderr="",
                exit_code=0,
                duration_s=duration,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            return RunnerResult(
                ok=False,
                stdout="",
                stderr="",
                exit_code=1,
                duration_s=duration,
                error=f"Health check raised an exception: {exc}",
            )

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
        """
        Grade the readability of `code` using the loaded HF model.

        Parameters
        ----------
        code            : Source code string to grade.
        tags            : Readability dimension tags from the config.
        language        : Programming language (e.g. "java", "python").
        rules           : Optional extra rules from the config.
        timeout_s       : Not enforced for transformers (blocking call).
        max_output_tokens: Overrides self.max_new_tokens when provided.
        """
        if max_output_tokens is not None:
            original = self.max_new_tokens
            self.max_new_tokens = max_output_tokens

        prompt = format_prompt(self.grade_prompt_template, tags, rules, language, code)

        try:
            raw_output = self._generate(prompt)
        except Exception as exc:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout="",
                raw_stderr="",
                parsed=None,
                error=f"Inference failed: {exc}",
                rationale="",
            )
        finally:
            if max_output_tokens is not None:
                self.max_new_tokens = original  # type: ignore[possibly-undefined]

        print(f"[DEBUG {self.name}] raw output: {repr(raw_output[:800])}", flush=True)

        parsed = extract_json_object(raw_output)
        if not parsed:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=raw_output,
                raw_stderr="",
                parsed=None,
                error="Could not find/parse JSON object in model output",
                rationale="",
            )

        score = clamp_score(parsed.get("score")) or compute_weighted_score(parsed)
        rationale = parsed.get("rationale", "")

        if score is None:
            return GradeResult(
                model_name=self.name,
                score=None,
                raw_stdout=raw_output,
                raw_stderr="",
                parsed=parsed,
                error="JSON parsed but 'score' missing or not numeric",
                rationale=rationale,
            )

        return GradeResult(
            model_name=self.name,
            score=score,
            raw_stdout=raw_output,
            raw_stderr="",
            parsed=parsed,
            error=None,
            rationale=rationale,
        )
