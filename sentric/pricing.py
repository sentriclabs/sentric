"""Built-in pricing table for common LLM models.

Prices are in USD per token (not per 1K tokens).
"""

# Pricing per token in USD
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
    "gpt-4": {"input": 30.00 / 1_000_000, "output": 60.00 / 1_000_000},
    "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
    "o1": {"input": 15.00 / 1_000_000, "output": 60.00 / 1_000_000},
    "o1-mini": {"input": 3.00 / 1_000_000, "output": 12.00 / 1_000_000},
    "o3-mini": {"input": 1.10 / 1_000_000, "output": 4.40 / 1_000_000},
    # Anthropic
    "claude-opus-4-20250514": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
    "claude-sonnet-4-20250514": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-haiku-3-5-20241022": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
}


def get_pricing(model_name: str, custom_pricing: dict | None = None) -> dict[str, float] | None:
    """Look up pricing for a model.

    Args:
        model_name: The model name to look up.
        custom_pricing: Optional custom pricing dict with "input" and "output" keys
            (USD per token). Overrides built-in pricing.

    Returns:
        Dict with "input" and "output" per-token prices, or None if unknown.
    """
    if custom_pricing is not None:
        return custom_pricing
    return MODEL_PRICING.get(model_name)


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, float],
) -> float:
    """Calculate cost in USD from token counts and pricing.

    Args:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        pricing: Dict with "input" and "output" per-token prices.

    Returns:
        Total cost in USD.
    """
    return input_tokens * pricing["input"] + output_tokens * pricing["output"]
