#!/usr/bin/env python3
"""Builds sample/cache from sample/catalog.json and sample/answers.json.

The sample's answers are written by hand in sample/answers.json. No model is
called. This script stores each one in the same shape the pipeline stores a
real answer, under the pipeline's own cache key for the current prompt, so
`run.py --replay` reads it like any stored answer. Each entry carries
"source": "scripted" and zero token counts so it can never pass for model
output. A cache key is a fingerprint of the model name, prompts, schema and
temperature. When the prompt changes, rerun this script to rebuild the cache.

    python3 tools/build_sample_cache.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cache import cache_key  # noqa: E402
from src.parsing import load_products  # noqa: E402
from src.prompts import SYSTEM_PROMPT, build_user_prompt, response_schema  # noqa: E402
from src.selector import DEFAULT_MODEL  # noqa: E402

CATALOG = ROOT / "sample" / "catalog.json"
ANSWERS = ROOT / "sample" / "answers.json"
CACHE_DIR = ROOT / "sample" / "cache"


def sample_key(product) -> str:
    user = build_user_prompt(product.product_title, product.product_description, product.search_results)
    return cache_key(DEFAULT_MODEL, SYSTEM_PROMPT, user, response_schema(False), 0.0)


def entries() -> dict[str, dict]:
    """Returns {cache key: stored payload} for every product with results."""
    products, errors = load_products(json.loads(CATALOG.read_text()))
    if errors:
        raise SystemExit(f"malformed catalog records: {errors}")
    answers = json.loads(ANSWERS.read_text())
    out = {}
    for p in products:
        if p.key not in answers:
            raise SystemExit(f"no scripted answer for {p.key!r} in sample/answers.json")
        out[sample_key(p)] = {
            "trusted_result_ids": answers[p.key]["trusted_result_ids"],
            "reason": None,
            "confidence": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "source": "scripted",
        }
    extra = set(answers) - {p.key for p in products}
    if extra:
        raise SystemExit(f"answers for products not in the catalog: {sorted(extra)}")
    return out


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    built = entries()
    for old in CACHE_DIR.glob("*.json"):
        if old.stem not in built:
            old.unlink()
    for key, payload in built.items():
        (CACHE_DIR / f"{key}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(built)} scripted entries to sample/cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
