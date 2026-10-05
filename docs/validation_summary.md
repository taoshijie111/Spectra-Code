# Data Validation Summary

The original fixed cipher has 20,005 entries with 20,005 distinct valid ten-bin, sum-nine codes. The QM9-derived table has 127,468 molecular rows and 3,220 distinct ten-bin codes. These two datasets share 2,270 code values.

The versioned dictionary contains all 20,005 original W2C word-code pairs exactly as released. The `original_code_10_9` field has ten nonnegative integers summing to nine; the separate `code` field has 400 nonnegative integers summing to 1,023. Labels, codes within each representation, and canonical molecular graphs are distinct. Its compressed SHA-256 digest is `f0b397aa90c12aed2fd2891baf52f738b5b714e4b4e2d6afefa271c0f4589b9a`.

For executable checks, run `python scripts/validate_release_assets.py`, `python scripts/validate_second_round.py`, and `python -m unittest discover -s tests` from the repository root.
