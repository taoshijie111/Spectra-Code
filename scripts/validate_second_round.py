"""Validate the versioned dictionary and supplementary result tables."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "second_round"
DICTIONARY = DATA / "versioned_dictionary_400_1023.jsonl.gz"
EXPECTED_SHA256 = "f0b397aa90c12aed2fd2891baf52f738b5b714e4b4e2d6afefa271c0f4589b9a"
TABLE_LENGTHS = {"s4": 13, "s5": 13, "s6a": 13, "s6b": 13,
                 "s7": 13, "s8": 7, "s9": 13, "s10": 13}


def normalize_word(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dictionary() -> dict[str, int | str]:
    assert sha256(DICTIONARY) == EXPECTED_SHA256
    with (ROOT / "data" / "data_word_mapping3_clean.json").open(encoding="utf-8") as handle:
        original = json.load(handle)
    original_by_word = {}
    for raw_code, raw_word in original.items():
        word = normalize_word(raw_word)
        code = ast.literal_eval(raw_code)
        assert len(code) == 10 and sum(code) == 9 and min(code) >= 0
        assert word not in original_by_word
        original_by_word[word] = tuple(code)
    assert len(original_by_word) == 20_005

    words, original_codes, codes, graphs, rows = set(), set(), set(), set(), set()
    with gzip.open(DICTIONARY, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            word = normalize_word(record["word"])
            original_code = record["original_code_10_9"]
            code = record["code"]
            assert record["word"] == word
            assert tuple(original_code) == original_by_word[word]
            assert len(code) == 400 and all(type(value) is int and value >= 0 for value in code)
            assert sum(code) == 1023
            assert isinstance(record["source_row"], int) and record["source_row"] >= 0
            assert isinstance(record["canonical_smiles"], str) and record["canonical_smiles"]
            assert re.fullmatch(r"[0-9a-f]{64}", record["public_member_sha256"])
            assert word not in words
            words.add(word)
            original_codes.add(tuple(original_code))
            codes.add(tuple(code))
            graphs.add(record["canonical_smiles"])
            rows.add(record["source_row"])
    assert words == original_by_word.keys()
    assert len(words) == len(original_codes) == len(codes) == len(graphs) == len(rows) == 20_005
    return {"entries": len(words), "original_code_length": 10, "original_code_total": 9,
            "expanded_code_length": 400, "expanded_code_total": 1023,
            "sha256": EXPECTED_SHA256}


def validate_tables() -> dict[str, int]:
    counts = {}
    for name, expected in TABLE_LENGTHS.items():
        with (DATA / f"table_{name}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == expected + 1, name
        width = len(rows[0])
        assert width > 1 and all(len(row) == width for row in rows), name
        assert len({row[0].casefold() for row in rows[1:]}) == expected, name
        counts[name] = expected

    with (DATA / "table_s4.csv").open(newline="", encoding="utf-8") as handle:
        capacity = {row["Design"]: row for row in csv.DictReader(handle)}
    with (DATA / "table_s5.csv").open(newline="", encoding="utf-8") as handle:
        recovery = {row["Design"]: row for row in csv.DictReader(handle)}
    assert int(capacity["b10_q9"]["All unique"]) == 3460
    assert int(capacity["b400_q1023"]["All unique"]) == 127441
    assert int(recovery["b10_q9"]["Source Top 10 of 1024"]) == 7
    assert int(recovery["b400_q1023"]["Source Top 10 of 1024"]) == 906
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps({"dictionary": validate_dictionary(), "table_rows": validate_tables()}, indent=2))


if __name__ == "__main__":
    main()
