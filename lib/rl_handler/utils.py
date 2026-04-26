from __future__ import annotations

import argparse
import re
from typing import Any, List

import torch

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in ("yes", "y", "true", "t"):
        return True
    if text in ("no", "n", "false", "f"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def ratio_0_1(value: Any) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Expected a float value.") from exc
    if ratio < 0.0 or ratio > 1.0:
        raise argparse.ArgumentTypeError("Expected a value in [0.0, 1.0].")
    return ratio


def parse_onnx_frontier_sizes(raw_value: Any) -> List[int]:
    def _flatten(value: Any) -> List[str]:
        if isinstance(value, bool):
            raise ValueError("Boolean is not a valid frontier-size value.")
        if isinstance(value, int):
            return [str(int(value))]
        if isinstance(value, float):
            if not float(value).is_integer():
                raise ValueError(f"Invalid frontier-size value '{value}'.")
            return [str(int(value))]
        if isinstance(value, (list, tuple, set)):
            tokens: List[str] = []
            for item in value:
                tokens.extend(_flatten(item))
            return tokens
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            return []
        return [tok for tok in re.split(r"[,\s]+", text) if tok]

    tokens = _flatten(raw_value)
    if not tokens:
        raise ValueError("At least one onnx frontier size is required.")

    sizes: List[int] = []
    seen = set()
    for token in tokens:
        try:
            size = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid frontier-size token '{token}'.") from exc
        if size <= 0:
            raise ValueError(f"onnx frontier size must be > 0, got {size}.")
        if size not in seen:
            seen.add(size)
            sizes.append(size)
    return sizes


def node_feature_dtypes_for_dataset(dataset_type: str) -> tuple[torch.dtype, str]:
    if str(dataset_type).upper() == "BITMASK":
        return torch.float32, "float32"
    return torch.int64, "int64"


def strip_quotes(value: Any) -> str:
    return str(value).replace('"', "").strip()


def parse_numeric_node_label_eval(node_obj: object) -> int:
    if isinstance(node_obj, bool):
        raise TypeError("Boolean node labels are not supported.")
    if isinstance(node_obj, int):
        return int(node_obj)
    if isinstance(node_obj, float):
        if not float(node_obj).is_integer():
            raise ValueError(f"Node label '{node_obj}' is not an integer.")
        return int(node_obj)
    if isinstance(node_obj, str):
        text = node_obj.strip()
        try:
            return int(text)
        except ValueError:
            parsed = float(text)
            if not parsed.is_integer():
                raise ValueError(f"Node label '{node_obj}' is not an integer.")
            return int(parsed)
    raise TypeError(f"Unsupported node label type: {type(node_obj)}")


def validate_int64_range_eval(value: int, *, context: str) -> int:
    if value < I64_MIN or value > I64_MAX:
        raise ValueError(f"{context} is out of int64 range [{I64_MIN}, {I64_MAX}].")
    return int(value)
