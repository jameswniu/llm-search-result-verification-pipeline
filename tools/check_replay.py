#!/usr/bin/env python3
"""Checks that sample/cache is the scripted cache for the current prompt.

The sample's answers are written by hand in sample/answers.json and turned
into cache files by tools/build_sample_cache.py. This check fails when a
cache file is missing, when one disagrees with sample/answers.json, when a
cache key no longer matches the current prompt (the cache must then be
rebuilt), when a replay of the sample misses, or when a cache file is an
orphan that no product uses. It needs no key and no openai package.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_sample_cache import CACHE_DIR, CATALOG, entries  # noqa: E402
from src.pipeline import run_file  # noqa: E402
from src.selector import SelectorConfig  # noqa: E402

REBUILD = "python3 tools/build_sample_cache.py"


def main() -> int:
    expected = entries()
    on_disk = {p.stem: p for p in CACHE_DIR.glob("*.json")} if CACHE_DIR.is_dir() else {}
    if not on_disk:
        print(f"FAIL: sample/cache is empty. Build it with: {REBUILD}")
        return 1

    problems = []
    missing = sorted(set(expected) - set(on_disk))
    orphans = sorted(set(on_disk) - set(expected))
    if missing and len(missing) == len(expected):
        problems.append(f"every key is missing. The prompt, schema, model or catalog changed, so the cache must be rebuilt with {REBUILD}")
    else:
        problems += [f"missing cache file for key {k[:12]}, rebuild with {REBUILD}" for k in missing]
    problems += [f"orphan cache file {k[:12]} is used by no product" for k in orphans]
    for key in sorted(set(expected) & set(on_disk)):
        stored = json.loads(on_disk[key].read_text())
        if stored != expected[key]:
            problems.append(f"cache file {key[:12]} disagrees with sample/answers.json, rebuild with {REBUILD}")

    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
        results, _, _ = run_file(
            input_path=CATALOG,
            output_path=Path(tmp) / "out.json",
            config=SelectorConfig(cache_dir=CACHE_DIR, replay=True),
        )
    hits = sum(1 for r in results if r.cache_hit)
    problems += [f"miss for {r.product_title!r}" for r in results if r.error]

    for p in problems:
        print(f"FAIL: {p}")
    if problems:
        return 1
    print(f"OK: {hits} of {len(expected)} scripted answers replayed, 0 misses, 0 orphan files, cache matches sample/answers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
