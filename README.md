# Spectra-Code

This repository currently contains two layers of material:

- a cleaned reviewer-facing release at the repository root
- the earlier `S2M/` subtree retained as legacy project code

The root-level release is the part intended to be shared first during peer review. It contains only small, deterministic assets that can be defended technically and uploaded to GitHub without pulling in large local research directories or demo-oriented scripts.

## Reviewer-Facing Release

Included now:

- `data/data_word_mapping3_clean.json`
  - cleaned word-to-code mapping used in the current working release
- `data/qm9_cond4.csv`
  - QM9-derived molecule table with `smiles` and `n_cond`
- `scripts/spectrum2code.py`
  - deterministic spectrum-to-code conversion
- `scripts/dft_log_utils.py`
  - deterministic Gaussian log parsing and IR broadening helpers
- `scripts/validate_release_assets.py`
  - validation script for the included assets
- `examples/mapping_qm9_exact_examples.csv`
  - small exact-match examples from the overlap of the mapping and QM9 code space
- `docs/release_scope.md`
  - inclusion and exclusion rationale
- `docs/validation_summary.md`
  - current validation findings
- `docs/license_status.md`
  - current public-license status note
- `requirements.txt`
  - minimal Python requirements for the included scripts

Quick validation:

```bash
python scripts/validate_release_assets.py
```

## What Is Not Included

This repository intentionally does not yet include:

- large local datasets such as `IR_broaden` and `20250103_unified_dataset`
- exploratory notebooks
- demo-oriented applications such as `app4.py`
- stochastic presentation scripts such as `enhanced_final_visualization.py`
- unrecovered or incomplete pipeline components

Large assets should be shared separately through a more appropriate channel if editors or reviewers request them.

## Current Limitation

This root-level release is a cleaned core asset package. It is not yet the final end-to-end reproducibility release for the full manuscript-wide C2S/S2M/reinforcement-learning training and inference pipeline.

## Legacy `S2M/` Subtree

The existing `S2M/` directory is preserved here as earlier project code. It has not been rewritten to match the cleaned reviewer-facing release and should be treated as legacy project material rather than the sole entry point for peer review.
