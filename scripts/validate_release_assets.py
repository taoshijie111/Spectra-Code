"""Validate the cleaned mapping and QM9 table included in reviewer_release."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from spectrum2code import process_spectrum_to_code


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_PATH = ROOT / "reports" / "asset_validation.json"


def validate_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    bad_keys = []
    bad_values = []
    codes = set()

    for key, value in mapping.items():
        try:
            parsed = ast.literal_eval(key)
        except Exception:
            bad_keys.append(key)
            continue

        if not isinstance(parsed, tuple) or len(parsed) != 10:
            bad_keys.append(key)
            continue
        if any((not isinstance(x, int)) or x < 0 for x in parsed):
            bad_keys.append(key)
            continue
        if sum(parsed) != 9:
            bad_keys.append(key)
            continue
        if not isinstance(value, str):
            bad_values.append(key)
            continue
        codes.add(parsed)

    return {
        "entries": len(mapping),
        "unique_codes": len(codes),
        "bad_keys": len(bad_keys),
        "bad_values": len(bad_values),
    }


def validate_qm9(path: Path) -> tuple[dict, set[tuple[int, ...]]]:
    df = pd.read_csv(path)
    bad_rows = 0
    codes = set()

    for value in df["n_cond"]:
        try:
            parsed = ast.literal_eval(value) if isinstance(value, str) else value
        except Exception:
            bad_rows += 1
            continue

        if not isinstance(parsed, list) or len(parsed) != 10:
            bad_rows += 1
            continue
        if any((not isinstance(x, int)) or x < 0 for x in parsed):
            bad_rows += 1
            continue
        if sum(parsed) != 9:
            bad_rows += 1
            continue
        codes.add(tuple(parsed))

    return {
        "rows": len(df),
        "unique_codes": len(codes),
        "bad_rows": bad_rows,
    }, codes


def validate_spectrum2code() -> dict:
    synthetic = [1.0] * 4000
    code = process_spectrum_to_code(synthetic)
    rejects_nonpositive_input = False

    try:
        process_spectrum_to_code([0.0] * 4000)
    except ValueError:
        rejects_nonpositive_input = True

    return {
        "output_length": len(code),
        "output_sum": sum(code),
        "all_integers": all(isinstance(x, int) for x in code),
        "rejects_nonpositive_input": rejects_nonpositive_input,
    }


def main() -> None:
    mapping_path = DATA_DIR / "data_word_mapping3_clean.json"
    qm9_path = DATA_DIR / "qm9_cond4.csv"

    mapping_result = validate_mapping(mapping_path)
    qm9_result, qm9_codes = validate_qm9(qm9_path)

    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    mapping_codes = {tuple(ast.literal_eval(key)) for key in mapping}

    report = {
        "mapping": mapping_result,
        "qm9": qm9_result,
        "spectrum2code": validate_spectrum2code(),
        "overlap": {
            "mapping_codes": len(mapping_codes),
            "qm9_codes": len(qm9_codes),
            "intersection": len(mapping_codes & qm9_codes),
            "mapping_codes_not_in_qm9": len(mapping_codes - qm9_codes),
            "qm9_codes_not_in_mapping": len(qm9_codes - mapping_codes),
        },
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
