"""Utility functions for langchain_luma."""

from typing import Any, Dict


def clean_none_values(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively remove None values from dictionary."""
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested = clean_none_values(v)
            if nested:
                clean[k] = nested
        elif v is not None:
            clean[k] = v
    return clean
