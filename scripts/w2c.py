"""Deterministic word-to-code lookup for the released lowercase cipher book."""

from __future__ import annotations

import argparse
import ast
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
DEFAULT_MAPPING = Path(__file__).resolve().parents[1] / "data" / "data_word_mapping3_clean.json"


def normalize_token(token: str) -> str:
    """Normalize an English token for exact lookup in the released vocabulary."""
    return unicodedata.normalize("NFKC", token).strip().casefold()


def tokenize(text: str) -> list[str]:
    """Extract English word tokens; punctuation and whitespace are delimiters."""
    normalized = unicodedata.normalize("NFKC", text)
    return [normalize_token(match.group(0)) for match in TOKEN_PATTERN.finditer(normalized)]


def load_word_to_code(mapping_path: str | Path) -> dict[str, tuple[int, ...]]:
    """Load the code-to-word JSON and invert it after enforcing one-to-one values."""
    with Path(mapping_path).open(encoding="utf-8") as handle:
        code_to_word = json.load(handle)
    if not isinstance(code_to_word, dict):
        raise ValueError("The cipher book must be a JSON object mapping codes to words")

    word_to_code: dict[str, tuple[int, ...]] = {}
    seen_codes: set[tuple[int, ...]] = set()
    for raw_code, raw_word in code_to_word.items():
        word = normalize_token(str(raw_word).strip())
        parsed_code = ast.literal_eval(raw_code)
        if not isinstance(parsed_code, (list, tuple)):
            raise ValueError(f"Invalid code for {raw_word!r}: {parsed_code!r}")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in parsed_code):
            raise ValueError(f"Code entries must be integers for {raw_word!r}: {parsed_code!r}")
        code = tuple(parsed_code)
        if len(code) != 10 or any(value < 0 or value > 9 for value in code) or sum(code) != 9:
            raise ValueError(f"Invalid code for {raw_word!r}: {code}")
        if word in word_to_code:
            raise ValueError(f"Duplicate normalized word: {word!r}")
        if code in seen_codes:
            raise ValueError(f"Duplicate spectral code: {code}")
        word_to_code[word] = code
        seen_codes.add(code)
    return word_to_code


def encode_tokens(
    tokens: Iterable[str],
    word_to_code: dict[str, tuple[int, ...]],
) -> list[dict[str, object]]:
    """Encode tokens without a silent fallback; OOV items remain explicitly unencoded."""
    result = []
    for raw_token in tokens:
        token = normalize_token(raw_token)
        code = word_to_code.get(token)
        result.append({
            "token": token,
            "status": "ok" if code is not None else "OOV",
            "code": list(code) if code is not None else None,
        })
    return result


def encode_text(text: str, mapping_path: str | Path) -> list[dict[str, object]]:
    return encode_tokens(tokenize(text), load_word_to_code(mapping_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact lookup in the released W2C cipher book")
    parser.add_argument("text", help="English text to encode")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="cipher-book JSON path")
    args = parser.parse_args()
    print(json.dumps(encode_text(args.text, args.mapping), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
