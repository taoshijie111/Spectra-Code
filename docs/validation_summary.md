# Validation Summary

Validation was run on the files included in this release.

## Checked Files

- `data/data_word_mapping3_clean.json`
- `data/qm9_cond4.csv`
- `scripts/spectrum2code.py`

## Current Findings

- `data_word_mapping3_clean.json`
  - 20005 entries
  - 20005 unique codes
  - every key parses to a 10-dimensional non-negative integer code with total sum 9
  - every value is a string
- `qm9_cond4.csv`
  - 127468 rows
  - 3220 unique `n_cond` codes
  - every parsed `n_cond` checked in validation is 10-dimensional, non-negative, and sums to 9
- code-space overlap
  - 2266 codes are shared between the cleaned mapping and QM9
  - 17739 mapping codes are not represented in QM9
  - 954 QM9 codes are not represented in the cleaned mapping

## Interpretation

The mapping file and QM9 table use the same code format, but they do not form a complete one-to-one system. The cleaned mapping defines a much larger word-to-code space, while the QM9 table provides molecule instances for only a subset of that space.

This is compatible with the included assets, but reviewer-facing documentation must not claim that every mapped word has an exact QM9 molecule.

## Small Safe Examples Included

`examples/mapping_qm9_exact_examples.csv` contains exact word-code-QM9 matches drawn only from the real intersection of these two assets.
