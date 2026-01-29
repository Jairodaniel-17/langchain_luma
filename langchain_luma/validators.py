# validators.py
import json

from langchain_luma.errors import PayloadTooLarge, ValidationError


def validate_id(id: str, max_len: int):
    if not id:
        raise ValidationError("id is required")
    if len(id) > max_len:
        raise ValidationError(f"id too long ({len(id)} > {max_len})")


def validate_vector(vector: list[float], max_dim: int):
    if not vector:
        raise ValidationError("vector is empty")
    if len(vector) > max_dim:
        raise ValidationError(f"vector too large ({len(vector)} > {max_dim})")


def validate_collection(name: str, max_len: int):
    if not name:
        raise ValidationError("collection name is required")
    if len(name) > max_len:
        raise ValidationError(f"collection name too long ({len(name)} > {max_len})")


def validate_k(k: int, max_k: int):
    if k <= 0:
        raise ValidationError("k must be > 0")
    if k > max_k:
        raise ValidationError(f"k too large ({k} > {max_k})")


def validate_json_size(obj, max_bytes: int, label: str):
    size = len(json.dumps(obj).encode("utf-8"))
    if size > max_bytes:
        raise PayloadTooLarge(f"{label} too large ({size} bytes > {max_bytes})")
