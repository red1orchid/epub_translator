"""Curated model lists per provider, with prices per 1M tokens.

The model IDs and fallback prices are hardcoded (verified against provider
docs as of 2026-08). Live prices are fetched from LiteLLM's community-
maintained price table and override the fallbacks when reachable, so
displayed prices stay up to date without code changes.
"""
from typing import Dict, List, Optional, Tuple

import requests

__all__ = ["MODELS", "FALLBACK_PRICES_DATE", "fetch_live_prices", "get_model_options"]

LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

FALLBACK_PRICES_DATE = "2026-08"

# (model_id, input $/1M tokens, output $/1M tokens) — fallback prices.
# First entry per provider is the default selection.
MODELS = {
    "openai": [
        ("gpt-5.1", 1.25, 10.0),
        ("gpt-5", 1.25, 10.0),
        ("gpt-5-mini", 0.25, 2.0),
        ("gpt-5-nano", 0.05, 0.40),
        ("gpt-4.1", 2.0, 8.0),
        ("gpt-4.1-mini", 0.40, 1.60),
        ("gpt-4o", 2.50, 10.0),
    ],
    "anthropic": [
        ("claude-sonnet-4-6", 3.0, 15.0),
        ("claude-sonnet-5", 3.0, 15.0),
        ("claude-haiku-4-5", 1.0, 5.0),
        ("claude-opus-5", 5.0, 25.0),
        ("claude-opus-4-8", 5.0, 25.0),
        ("claude-opus-4-6", 5.0, 25.0),
    ],
}


def fetch_live_prices(timeout: float = 4.0) -> Dict[str, Tuple[float, float]]:
    """Return {model_key: ($/1M input, $/1M output)} from the LiteLLM table.

    Raises on network/parse errors — callers should catch and fall back.
    """
    resp = requests.get(LITELLM_PRICES_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    prices = {}
    for key, info in data.items():
        if not isinstance(info, dict):
            continue
        cin = info.get("input_cost_per_token")
        cout = info.get("output_cost_per_token")
        if isinstance(cin, (int, float)) and isinstance(cout, (int, float)):
            prices[key] = (cin * 1_000_000, cout * 1_000_000)
    return prices


def _fmt_price(value: float) -> str:
    s = f"${value:.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def get_model_options(
    provider_name: str,
    live_prices: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[dict]:
    """Model options for a provider: [{id, label}], label includes prices."""
    options = []
    for model_id, fallback_in, fallback_out in MODELS.get(provider_name, []):
        price_in, price_out = fallback_in, fallback_out
        if live_prices:
            for key in (model_id, f"{provider_name}/{model_id}"):
                if key in live_prices:
                    price_in, price_out = live_prices[key]
                    break
        options.append({
            "id": model_id,
            "label": f"{model_id} · {_fmt_price(price_in)} in / {_fmt_price(price_out)} out per 1M tokens",
        })
    return options
