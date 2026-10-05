# Data Validation Summary

The original fixed cipher has 20,005 entries with 20,005 distinct valid ten-bin, sum-nine codes. The QM9-derived table has 127,468 molecular rows and 3,220 distinct ten-bin codes. These two datasets share 2,270 code values.

The expanded 400-bin dictionary has 20,005 distinct normalized labels, codes, and canonical molecular graphs. Every code is a nonnegative integer vector of length 400 with total 1,023. Its compressed SHA-256 digest is `403c4fbd034ff776f9a2d138a9a91298b976e9c9ec8a69790876c7756dfa2285`.

For executable checks, run `python scripts/validate_release_assets.py`, `python scripts/validate_second_round.py`, and `python -m unittest discover -s tests` from the repository root.
