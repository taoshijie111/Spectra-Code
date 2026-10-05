"""Integer spectrum encoders used in the expanded-bin comparison."""

from __future__ import annotations

from math import comb

import numpy as np


def equal_edges(bins: int, length: int = 4000) -> np.ndarray:
    if not 1 <= bins <= length:
        raise ValueError("bins must be between 1 and spectrum length")
    return np.rint(np.linspace(0, length, bins + 1)).astype(np.int64)


def adaptive_edges(training_spectra: np.ndarray, bins: int = 10) -> np.ndarray:
    spectra = np.atleast_2d(np.asarray(training_spectra, dtype=np.float64))
    if not 1 <= bins <= spectra.shape[1]:
        raise ValueError("invalid bin count")
    mean = spectra.mean(axis=0)
    cumulative = np.cumsum(np.maximum(mean, 0))
    if not np.isfinite(cumulative).all() or cumulative[-1] <= 0:
        raise ValueError("training spectra must have positive finite mass")
    interior = np.searchsorted(cumulative, np.arange(1, bins) * cumulative[-1] / bins) + 1
    edges = np.r_[0, interior, len(mean)].astype(np.int64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("degenerate adaptive boundaries")
    return edges


def bin_masses(spectra: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(np.asarray(spectra))
    boundaries = np.asarray(edges, dtype=np.int64)
    if boundaries[0] != 0 or boundaries[-1] != values.shape[1] or np.any(np.diff(boundaries) <= 0):
        raise ValueError("edges must partition the spectrum")
    return np.stack(
        [values[:, start:end].sum(axis=1, dtype=np.float64)
         for start, end in zip(boundaries[:-1], boundaries[1:])],
        axis=1,
    )


def quantize(masses: np.ndarray, total: int = 9) -> np.ndarray:
    values = np.atleast_2d(np.asarray(masses, dtype=np.float64))
    if not 0 <= total <= np.iinfo(np.int16).max:
        raise ValueError("total is outside the supported integer range")
    if np.any(values < -1e-9) or np.any(values.sum(axis=1) <= 0) or not np.isfinite(values).all():
        raise ValueError("spectrum mass must be finite and positive")
    scaled = np.maximum(values, 0) / values.sum(axis=1, keepdims=True) * total
    codes = np.rint(scaled).astype(np.int16)
    residual = total - codes.sum(axis=1)
    for row in np.flatnonzero(residual):
        remaining = int(residual[row])
        order = np.argsort(-scaled[row] if remaining > 0 else scaled[row], kind="stable")
        for column in order:
            if remaining == 0:
                break
            step = 1 if remaining > 0 else -1
            if 0 <= codes[row, column] + step <= total:
                codes[row, column] += step
                remaining -= step
        if remaining:
            raise AssertionError("integer total was not corrected")
    return codes


def encode(spectra: np.ndarray, edges: np.ndarray, total: int = 9) -> np.ndarray:
    return quantize(bin_masses(spectra, edges), total)


def occupancy(codes: np.ndarray, total: int) -> dict[str, float | int]:
    values = np.atleast_2d(np.asarray(codes))
    if len(values) < 2 or np.any(values < 0) or np.any(values.sum(axis=1) != total):
        raise ValueError("at least two valid fixed-total codes are required")
    _, counts = np.unique(values, axis=0, return_counts=True)
    population = len(values)
    probabilities = counts / population
    return {
        "n_records": population,
        "unique_codes": len(counts),
        "bins": values.shape[1],
        "total": total,
        "theoretical_capacity": comb(total + values.shape[1] - 1, values.shape[1] - 1),
        "entropy_bits": float(-(probabilities * np.log2(probabilities)).sum()),
        "pair_collision_probability": float((counts * (counts - 1)).sum() / (population * (population - 1))),
    }
