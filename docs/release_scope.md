# Release Scope

This note records what was included, what was excluded, and why.

## Must-Have Files Included Now

- `data/data_word_mapping3_clean.json`
- `data/qm9_cond4.csv`
- `scripts/spectrum2code.py`
- `scripts/dft_log_utils.py`
- `scripts/validate_release_assets.py`
- `examples/mapping_qm9_exact_examples.csv`
- `README.md`
- `requirements.txt`

## Can Be Shared Later, But Not In This Package

- `IR_broaden`
  - only needed if a deterministic figure or retrieval script that consumes those text spectra is restored to reviewer scope
- `20250103_unified_dataset`
  - large processed dataset directory; not included because the currently selected release does not expose those visualization scripts
- raw Gaussian log directories such as `all_logs0527`, `all_logs0529`, and related folders
  - useful for deeper provenance, but too large for the present GitHub package
- `experiment_spectral_code_final3.csv`
  - useful for dataset reconstruction, but too large for this first release package

## Explicitly Excluded

- `app4.py`
  - demo-oriented app with heuristic and random candidate selection
- `enhanced_final_visualization.py`
  - stochastic presentation script
- `figure1_replot_10.py`
  - deterministic, but excluded by package choice
- `vae_plot_0811.ipynb`
- `figure/plot.ipynb`
- `build_data_word_mapping3_clean.py`
- `data_word_mapping3_clean_report.md`
- historical manuscript, SI, and cover-letter documents
- old mapping variants such as `data_word_mapping3 2.json`

## Excluded From Related Workspaces After Review

- root and dataset copies of `create_unified_dataset.py`
  - not included because the QM9-only branch is unfinished and contains explicit placeholder logic
- `scientific_baseline_correction.py`
  - not included because it recalculates codes with a different rule from the main 10-bin sum-to-9 code system
- `check_qm9_molecules.py` and `analyze_and_select_molecules.py`
  - auxiliary analysis utilities, not central to reviewer-facing reproduction
- `log24000.py`
  - original file contains Windows-only save paths and noise utilities; a cleaned deterministic subset was extracted into `scripts/dft_log_utils.py`

## Working Rule Used For Inclusion

A file was included only if it satisfied all of the following:

- deterministic output under the selected runtime path
- no unresolved TODO that blocks interpretation
- no dependence on unrecovered local outputs
- no obvious demo-only or presentation-only behavior
- small enough to be GitHub-friendly

## Remaining Gap

After reviewing the current and related local workspaces, no clean, self-contained, reviewer-ready implementation of the full manuscript-wide C2S/S2M/reinforcement-learning training-and-inference pipeline was found.

This package should therefore be described as a cleaned core asset release, not as the final complete reproducibility repository for every central manuscript claim.
