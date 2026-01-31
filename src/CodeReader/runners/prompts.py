DEFAULT_HEALTH_PROMPT = "Reply with exactly: OK"

DEFAULT_GRADE_PROMPT = """
    You are a strict readability grader.
    Rules: \n
        - Use the tags to know the context and what to focus on.\n
        - Use the Language to know what programming language to grade it in.\n
        - Grade the code of how readable it is.\n\n
    Return JSON ONLY (no markdown, no extra text), exactly in this schema:
    {{"score": <integer 0-100>, "rationale": "<short explanation>"}}
    \n\n
    Tags: {tags}\n
    Language: {language}
    \n\n
    Code:\n
    ```{language}\n
    {code}
    ```
"""
