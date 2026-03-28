"""Deterministic Gaussian log parsing and IR broadening helpers."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import wofz


def _to_float_list(values: Sequence[float] | str) -> List[float]:
    if isinstance(values, str):
        parsed = ast.literal_eval(values)
    else:
        parsed = values
    return [float(v) for v in parsed]


def parse_gaussian_log(logfile: str | Path) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    path = Path(logfile)
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None, None

    if not any("Frequencies" in line for line in lines):
        return None, None

    frequencies: List[float] = []
    intensities: List[float] = []
    try:
        for line in lines:
            match_freq = re.search(r"Frequencies\s*--\s*([\d\.\s\-]+)", line)
            match_int = re.search(r"IR Inten\s*--\s*([\d\.\s\-]+)", line)
            if match_freq:
                frequencies.extend(float(x) for x in match_freq.group(1).split())
            if match_int:
                intensities.extend(float(x) for x in match_int.group(1).split())
    except Exception:
        return None, None

    if not frequencies or not intensities:
        return None, None

    size = min(len(frequencies), len(intensities))
    return frequencies[:size], intensities[:size]


def voigt_profile(x: np.ndarray, amplitude: float, position: float, fwhm: float, shape: float = 1.0) -> np.ndarray:
    scale = 1 / wofz(np.zeros(len(x)) + 1j * np.sqrt(np.log(2.0)) * shape).real
    return scale * amplitude * wofz(
        2 * np.sqrt(np.log(2.0)) * (x - position) / fwhm + 1j * np.sqrt(np.log(2.0)) * shape
    ).real


def broaden_ir_spectrum(
    frequencies: Sequence[float] | str,
    intensities: Sequence[float] | str,
    spectrum_length: int = 4000,
    broaden_factor: float = 20.0,
    shape: float = 1.0,
) -> np.ndarray:
    freq_list = _to_float_list(frequencies)
    int_list = _to_float_list(intensities)

    if len(freq_list) != len(int_list):
        size = min(len(freq_list), len(int_list))
        freq_list = freq_list[:size]
        int_list = int_list[:size]

    x = np.arange(spectrum_length, dtype=float)
    y = np.zeros(spectrum_length, dtype=float)
    for freq, intensity in zip(freq_list, int_list):
        y += voigt_profile(x, amplitude=float(intensity), position=float(freq), fwhm=broaden_factor, shape=shape)
    return y
