"""Tests for the portable integer spectrum encoders."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from expanded_encoding import adaptive_edges, encode, equal_edges, occupancy  # noqa: E402
from spectrum2code import process_spectrum_to_code  # noqa: E402


class ExpandedEncodingTests(unittest.TestCase):
    def test_original_ten_bin_parity(self):
        rng = np.random.default_rng(211)
        spectra = rng.random((20, 4000))
        current = encode(spectra, equal_edges(10), 9)
        original = np.asarray([process_spectrum_to_code(row) for row in spectra])
        np.testing.assert_array_equal(current, original)

    def test_expanded_codes_and_capacity(self):
        spectra = np.vstack((np.arange(1, 4001), np.arange(4000, 0, -1)))
        codes = encode(spectra, equal_edges(400), 1023)
        self.assertEqual(codes.shape, (2, 400))
        np.testing.assert_array_equal(codes.sum(axis=1), [1023, 1023])
        self.assertEqual(occupancy(codes, 1023)["unique_codes"], 2)

    def test_training_fitted_edges(self):
        training = np.vstack((np.arange(1, 4001), np.arange(4000, 0, -1)))
        edges = adaptive_edges(training, 10)
        self.assertEqual(edges[0], 0)
        self.assertEqual(edges[-1], 4000)
        self.assertTrue(np.all(np.diff(edges) > 0))

    def test_rejects_zero_mass(self):
        with self.assertRaises(ValueError):
            encode(np.zeros((1, 4000)), equal_edges(10), 9)


if __name__ == "__main__":
    unittest.main()
