# Spectra-Code

This repository provides the word-to-code dictionary, spectrum-encoding utilities, molecular data table, and supplementary benchmark results for *The Natural Coding of Language: SPECTRA is All You Need*.

## Data

- [`data/data_word_mapping3_clean.json`](data/data_word_mapping3_clean.json): the fixed 20,005-entry word-to-code cipher used by the original ten-bin, sum-nine workflow.
- [`data/qm9_cond4.csv`](data/qm9_cond4.csv): 127,468 QM9-derived molecular records with ten-bin codes.
- [`examples/mapping_qm9_exact_examples.csv`](examples/mapping_qm9_exact_examples.csv): exact-code examples connecting the fixed dictionary and QM9 table.
- [`data/second_round/versioned_dictionary_400_1023.jsonl.gz`](data/second_round/versioned_dictionary_400_1023.jsonl.gz): a separate 20,005-entry assignment for the 400-bin, sum-1,023 representation. The original cipher remains unchanged.
- [`data/second_round/table_s4.csv`](data/second_round/table_s4.csv) through [`table_s10.csv`](data/second_round/table_s10.csv): the per-design capacity, collision, reconstruction, molecular-readout, perturbation, and resolution results reported in Supplementary Tables S4-S10. See the [table guide](data/second_round/README.md).

## Code

- [`scripts/w2c.py`](scripts/w2c.py): exact lookup in the original fixed vocabulary.
- [`scripts/spectrum2code.py`](scripts/spectrum2code.py): original ten-bin spectrum-to-code conversion.
- [`scripts/expanded_encoding.py`](scripts/expanded_encoding.py): equal-width and training-fitted adaptive binning, fixed-total quantization, and occupancy statistics for the expanded-code comparison.
- [`scripts/dft_log_utils.py`](scripts/dft_log_utils.py): Gaussian frequency-log parsing and IR broadening utilities.
- [`scripts/validate_release_assets.py`](scripts/validate_release_assets.py) and [`scripts/validate_second_round.py`](scripts/validate_second_round.py): checks for the released data assets and supplementary results.
- [`S2M/`](S2M/): spectrum-to-molecule source modules and model configurations.

## Quick Checks

Install the packages in [`requirements.txt`](requirements.txt), then run:

```bash
python scripts/validate_release_assets.py
python scripts/validate_second_round.py
python -m unittest discover -s tests
```

The [data and methods index](docs/release_scope.md) describes the relationship between the original cipher and the expanded comparison. Repository access and licensing terms are in [`NOTICE.md`](NOTICE.md).
