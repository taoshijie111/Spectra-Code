# Spectra-Code

This repository contains:

- a reviewer-facing release at the repository root
- an archived `S2M/` source subtree retained only for historical context

For peer review, use the repository root only. The root-level release is the maintained reviewer-facing entry point and contains the deterministic assets that are currently intended to be shared first.

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
- `NOTICE.md`
  - repository-level notice describing the current public-access and licensing status
- `requirements.txt`
  - minimal Python requirements for the included scripts

Quick validation:

```bash
python scripts/validate_release_assets.py
```

Reviewer entry point:

- start from this `README.md`
- use the root-level `data/`, `scripts/`, `examples/`, and `docs/` directories
- read `NOTICE.md` for the current repository-wide licensing status
- treat `S2M/` as archived background code, not as the primary reviewer workflow

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

The `S2M/` directory is preserved only as archived source/config material from an earlier project state.

- it is not the primary reviewer-facing entry point
- historical notebooks, training logs, and generated figures have been removed from the public repository to reduce ambiguity
- if material from `S2M/` is needed for peer review, the reviewer-facing root documentation should take precedence
