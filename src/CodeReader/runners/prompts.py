DEFAULT_HEALTH_PROMPT = "Reply with exactly: OK"

DEFAULT_GRADE_PROMPT = """
    You are a strict readability grader.
    Return JSON ONLY (no markdown, no extra text), exactly in this schema:
    {{"score": <integer 0-100>, "rationale": "<short explanation>"}}

    Tag: {tag}
    Language: {language}

    Code:
    ```{language}
    {code}
    ```
"""
