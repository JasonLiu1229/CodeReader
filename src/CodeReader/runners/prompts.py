DEFAULT_HEALTH_PROMPT = "Reply with exactly: OK"

DEFAULT_GRADE_PROMPT = """
    You are a strict readability grader.
    Use the tags to know the context.
    Use the Language to know what programming language to grade it in.
    Return JSON ONLY (no markdown, no extra text), exactly in this schema:
    {{"score": <integer 0-100>, "rationale": "<short explanation>"}}

    Tags: {tags}
    Language: {language}

    Code:
    ```{language}
    {code}
    ```
"""
