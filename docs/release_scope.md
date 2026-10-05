# Data and Methods Index

## Original Cipher

The fixed W2C dictionary is `data/data_word_mapping3_clean.json`. It contains 20,005 normalized labels assigned to distinct ten-component nonnegative integer codes, each summing to nine. The formal alphabet has `C(18, 9) = 48,620` codes. `scripts/w2c.py` performs exact lookup, and `scripts/spectrum2code.py` converts a 4,000-point spectrum to the same ten-bin code format.

`data/qm9_cond4.csv` is a separate molecular table containing 127,468 records and their ten-bin codes. Dictionary membership and the occurrence of a code in a particular molecular dataset are different properties.

## Expanded Comparison

The expanded-encoding analysis compares eleven integer representations and two PCA/vector-quantization controls on a graph-disjoint split of 127,468 computed spectra: 102,123 training, 12,794 validation, and 12,551 test records. `scripts/expanded_encoding.py` implements the integer encoding and occupancy calculations.

The versioned dictionary in `data/second_round/` includes all 20,005 original 10/9 W2C word-code pairs unchanged, alongside separate 400-bin, sum-1,023 computed-spectrum codes and canonical molecular graphs for those labels. The original ten-bin pairs remain authoritative for W2C and the manuscript's case studies; the expanded codes are a separately versioned comparison, not replacements.

Supplementary Tables S4-S10 are distributed as CSV in `data/second_round/`. They report formal capacity in bits, observed occupancy, entropy, pair-collision probability, reconstruction, source-graph recovery, and matched perturbation and resolution results. The [table guide](../data/second_round/README.md) gives denominators and endpoint definitions.

## Validation

Run `python scripts/validate_release_assets.py` for the original assets, `python scripts/validate_second_round.py` for the versioned dictionary and tables, and `python -m unittest discover -s tests` for the portable encoding functions.
