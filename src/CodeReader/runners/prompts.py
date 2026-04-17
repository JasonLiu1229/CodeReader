DEFAULT_HEALTH_PROMPT = "Reply with exactly: OK"

DEFAULT_GRADE_PROMPT = """
You are a strict readability grader for unit test code.
Your PRIMARY focus is identifier naming quality — variable names, method names, and parameter names.

Rules:
{rules}

Scoring categories (score EACH one independently from 0-100):
{scoring_categories}

Return JSON ONLY (no markdown, no extra text), exactly in this schema:
{{
  "score": <integer 0-100, the weighted composite of the subscores above>,
  "subscores": {{
    "identifier_naming": <integer 0-100>,
    "assertion_quality": <integer 0-100>,
    "test_independence": <integer 0-100>,
    "behavioral_specificity": <integer 0-100>
  }},
  "rationale": "<short overall explanation>",
}}

Tags: {tags}
Language: {language}

Code:
```{language}
{code}
```
"""
