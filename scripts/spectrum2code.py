"""Deterministic utilities for converting a 4000-point spectrum to a 10-bin code."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def integrate_data(data: Iterable[float], window_size: int = 400) -> List[float]:
    values = list(data)
    return [float(np.sum(values[i:i + window_size])) for i in range(0, len(values), window_size)]


def normalize_to_integer_list(
    values: Iterable[float],
    target_sum: int = 9,
    max_value: int = 9,
) -> List[int]:
    numeric_values = [float(v) for v in values]
    total = sum(numeric_values)
    if total <= 0:
        raise ValueError("Spectrum must have positive total intensity")

    normalized = [(val / total) * max_value for val in numeric_values]
    integers = [int(round(val)) for val in normalized]

    difference = target_sum - sum(integers)
    if difference != 0:
        order = sorted(
            range(len(normalized)),
            key=lambda idx: normalized[idx],
            reverse=(difference > 0),
        )
        for idx in order:
            if difference == 0:
                break
            step = 1 if difference > 0 else -1
            new_value = integers[idx] + step
            if 0 <= new_value <= max_value:
                integers[idx] = new_value
                difference -= step

    if len(integers) != 10:
        raise ValueError(f"Expected 10 code bins, got {len(integers)}")
    if sum(integers) != target_sum:
        raise ValueError(f"Code sum is {sum(integers)}, expected {target_sum}")
    if any(v < 0 or v > max_value for v in integers):
        raise ValueError("Code value out of range")

    return integers


def process_spectrum_to_code(ir_spectrum_data: Iterable[float]) -> List[int]:
    integrated_values = integrate_data(ir_spectrum_data, window_size=400)
    return normalize_to_integer_list(integrated_values, target_sum=9, max_value=9)
