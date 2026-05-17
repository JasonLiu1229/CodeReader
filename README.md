# CodeReader

CodeReader is a **local, LLM-based code readability grader**. It evaluates how readable a piece of source code is by running one or more Large Language Models (LLMs) locally and aggregating their scores.

This tool is designed primarily for **research and experimentation**, especially in the context of evaluating readability, naming quality, and structural clarity of code using LLMs instead of fixed syntactic metrics.

---

## Features

- **Readability scoring (0–100)** using one or more LLMs
- **Weighted averages** across multiple models
- **Model rationales** explaining _why_ a score was given
- **Tag-based evaluation** (e.g. identifiers, structure, comments)
- **Rule-based evaluation** — add custom rules on top of tags
- **Structured logging** of all grading results
- **Fully local execution** (no cloud APIs required)
- **OpenAI support** for API-based grading (requires an API key)
- **HuggingFace Transformers support** for local model inference, with optional LoRA/PEFT adapter loading
- **Configurable via YAML** (models, weights, prompts, tags)

---

## How it works (high-level)

1. You provide a piece of code (file, inline text, or stdin)
2. A YAML configuration specifies:
   - which LLMs to use
   - their weights
   - what aspects of readability to evaluate
3. CodeReader sends a structured prompt to each model
4. Each model returns a JSON score + rationale
5. CodeReader aggregates the results and prints a table
6. Results are appended to a log file for later analysis

---

## Installation

### Requirements

- Python **3.12+**
- A supported runner backend (see below)

### Install from source (development)

```bash
poetry install
```

### Install via pip (once published)

```bash
pip install codereader
```

### Optional extras

CodeReader supports multiple backends. Only install what you need — heavy dependencies like `torch` are **never installed by default**.

| Extra          | What it adds                                         | Install command                        |
| -------------- | ---------------------------------------------------- | -------------------------------------- |
| _(none)_       | Ollama runner only                                   | `pip install codereader`               |
| `transformers` | HuggingFace Transformers + LoRA/PEFT adapter support | `pip install codereader[transformers]` |

The `transformers` extra pulls in `torch`, `transformers`, `peft`, and `accelerate`. If you try to use `TransformersRunner` without installing this extra, CodeReader will raise a clear error with the exact install command.

---

## Runner backends

### Ollama

Requires [Ollama](https://ollama.com) installed and running locally, with at least one model pulled (e.g. `qwen2.5-coder`, `deepseek-coder`).

```bash
ollama serve
ollama pull qwen2.5-coder
```

### HuggingFace Transformers

Requires the `transformers` extra (see above). Supports any causal LM on the HuggingFace Hub or a local path, with optional LoRA / PEFT adapter weights.

```bash
pip install codereader[transformers]
```

---

## Usage

CodeReader exposes a CLI called `codereader`.

### Basic command

```bash
codereader grade -c config.yml -f example.py
```

### Input methods

Exactly **one** of the following must be provided:

- `--file / -f` – path to a source file
- `--text` – inline source code as a string
- `--stdin` – read code from standard input

Examples:

```bash
codereader grade -c config.yml --text "int x = 0;"
```

```bash
cat example.py | codereader grade -c config.yml --stdin
```

### Useful options

- `--name` – override filename label in logs
- `--quiet` – suppress console output (logging still happens)
- `--simple` – simplified console output (quiet takes priority over simple)

---

## Output

The CLI prints a table with:

- Model name
- Score (0–100)
- Weight
- Rationale
- Error (if any)

Below the table, CodeReader prints:

- **Average score**
- **Weighted average score**

All results are also appended to a log file specified in the config.

---

## Configuration (YAML)

CodeReader is fully driven by a YAML config file.

Typical configuration sections:

- `language` – programming language of the code
- `tags` – aspects of readability to evaluate
- `models` – list of LLM runners and weights
- `settings` – logging paths and runtime settings

### Ollama runner example

```yaml
language: java
tags:
  - identifiers
  - structure

models:
  - name: qwen
    type: ollama
    model: qwen2.5-coder
    weight: 1.0
    runner_config:
      no_think: true # disables deep thinking to shorten output

settings:
  log_path: readability_log.txt
```

### Transformers runner example

```yaml
language: java
tags:
  - identifiers
  - structure

models:
  - name: mistral-finetuned
    type: transformers
    model: mistralai/Mistral-7B-Instruct-v0.2 # HF model ID or local path
    weight: 1.0
    runner_config:
      adapter_path: ./checkpoints/lora-epoch3 # optional: LoRA/PEFT adapter dir
      device: auto # auto | cuda | cpu | mps
      torch_dtype: float16 # float32 | float16 | bfloat16
      max_new_tokens: 512
      temperature: 0.1

settings:
  log_path: readability_log.txt
```

`adapter_path` is optional. When provided, the LoRA weights are merged into the base model before inference (no runtime overhead). Omit it to run the base model as-is.

### Mixed runners example

You can combine multiple backends in a single config and aggregate their scores with weights:

```yaml
language: python
tags:
  - identifiers
  - comments

models:
  - name: qwen-ollama
    type: ollama
    model: qwen2.5-coder
    weight: 0.5
    runner_config:
      no_think: true

  - name: mistral-hf
    type: transformers
    model: mistralai/Mistral-7B-Instruct-v0.2
    weight: 0.5
    runner_config:
      device: auto
      torch_dtype: bfloat16

settings:
  log_path: readability_log.txt
```

---

## Logging

Each grading run appends a structured entry to the log file, including:

- filename
- individual model scores and rationales
- averages

Useful for **dataset-level analysis**, benchmarking, and research experiments.

---

## Research context

CodeReader was developed as part of a **master's thesis** exploring:

> _Renaming identifiers of unit-tests generated by automated testing using LLMs_

---

## License

This project is licensed under the **GNU General Public License v3 (GPLv3)**.

- You are free to use, modify, and redistribute the software
- Derivative works must also be released under GPLv3

See the `LICENSE` file for details.

---

## Contributing

Contributions are welcome, especially around:

- additional model runners
- prompt engineering
- evaluation methodology

Please open an issue or pull request.

---

## Roadmap / Future work

CodeReader is an active research project. Planned and in-progress work includes:

- Additional LLM backends
  - More API-based LLMs (e.g. Anthropic, or generic OpenAI-compatible endpoints)
  - Batch / dataset-level grading
  - Cached or replay-based evaluation for reproducibility

- Improved health checks and timeout handling per runner type

- Expanded template system
  - More language-specific prompt templates (Java, Python, C/C++, Rust, etc.)
  - Templates targeting specific readability dimensions:
    - identifier naming
    - test code readability
    - control-flow complexity
    - documentation and comments
  - Easier authoring and validation of custom prompt templates

- Analysis & reporting
  - Richer logging formats (JSON / CSV export)
  - Dataset-level summaries and comparisons
  - Inter-model agreement and variance analysis

---

## Status

This project is **research-oriented** and under active development.
Expect breaking changes before a stable 1.0 release.
