from __future__ import annotations

from typing import Any, Dict, List, Optional


DEFAULT_WEIGHTS: Dict[str, float] = {
    "identifier_naming": 0.50,
    "assertion_quality": 0.20,
    "test_independence": 0.15,
    "behavioral_specificity": 0.15,
}


_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "identifier_naming": (
        "Quality of all identifiers (variables, parameters, method names). "
        "Score 0 if names are generic auto-generated tokens (var0, obj1, int0, string0). "
        "Score 100 if every name clearly conveys its role and intent."
    ),
    "assertion_quality": (
        "Expressiveness of assertions. Prefer assertEquals/assertNotNull/assertThrows "
        "over assertTrue(a == b). Penalise missing messages when failure reason is non-obvious."
    ),
    "test_independence": (
        "Whether the test is self-contained with no shared mutable state or "
        "dependency on execution order."
    ),
    "behavioral_specificity": (
        "Whether the test name and body together make the tested behaviour and "
        "expected outcome obvious without reading production code."
    ),
}


def _build_scoring_categories_str(weights: Dict[str, float]) -> str:
    lines = []
    for category, weight in weights.items():
        desc = _CATEGORY_DESCRIPTIONS.get(category, "")
        lines.append(f"- {category} (weight {weight:.0%}): {desc}")
    return "\n".join(lines)


def format_prompt(
    template: str,
    tags: List[str],
    rules: Optional[List[str]],
    language: str,
    code: str,
    weights: Optional[Dict[str, float]] = None,
) -> str:
    """
    Format the grade prompt.

    Backward-compatible with the original signature:
      format_prompt(template, tags, rules, language, code)

    The optional `weights` parameter lets you pass scoring weights from the
    config; if omitted, DEFAULT_WEIGHTS are used.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    tags_str = ", ".join(tags)
    rules_str = (
        "\n".join(f"- {r}" for r in rules) if rules else "No specific rules specified."
    )
    scoring_categories_str = _build_scoring_categories_str(weights)

    return template.format(
        tags=tags_str,
        language=language,
        code=code,
        rules=rules_str,
        scoring_categories=scoring_categories_str,
    )


def compute_weighted_score(
    parsed: Optional[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[int]:
    """
    Given the parsed JSON dict returned by the LLM, recompute the final score
    from the subscores using Python-side weights.

    Returns None if subscores are absent (falls back to the LLM's own 'score').

    Usage in runners — after you have `parsed = extract_json_object(content)`:

        from .utils import compute_weighted_score
        score = compute_weighted_score(parsed, weights) or clamp_score(parsed.get("score"))
    """
    if not parsed:
        return None

    if weights is None:
        weights = DEFAULT_WEIGHTS

    subscores: Dict[str, Any] = parsed.get("subscores") or {}
    if not subscores:
        return None

    total_weight = 0.0
    weighted_sum = 0.0
    for category, weight in weights.items():
        raw = subscores.get(category)
        if raw is None:
            continue
        try:
            value = max(0, min(100, int(round(float(raw)))))
        except (TypeError, ValueError):
            continue
        weighted_sum += value * weight
        total_weight += weight

    if total_weight == 0:
        return None

    return int(round(weighted_sum / total_weight))
