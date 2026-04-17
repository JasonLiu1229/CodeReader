DEFAULT_HEALTH_PROMPT = "Reply with exactly: OK"

DEFAULT_GRADE_PROMPT = """
You are a strict readability grader for unit test code.

Rules:
{rules}

Scoring categories (score EACH one independently from 0-100):
{scoring_categories}

Return JSON ONLY. No markdown. No code fences. No explanation before or after. No extra text of any kind.
The "rationale" field inside the JSON is the only place for explanation — put everything there.
Output must start with {{ and end with }} and contain nothing else.

Exactly this schema:
{{
  "score": <integer 0-100, the weighted composite of the subscores above>,
  "subscores": {{
    "identifier_naming": <integer 0-100>,
    "assertion_quality": <integer 0-100>,
    "test_independence": <integer 0-100>,
    "behavioral_specificity": <integer 0-100>
  }},
  "rationale": "<explanation of scores — mention specific identifier names that are good or bad>"
}}

Tags: {tags}
Language: {language}

Code:
```{language}
{code}
```
"""
