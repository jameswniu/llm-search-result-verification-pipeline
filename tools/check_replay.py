#!/usr/bin/env python3
"""Checks that sample/cache holds one recorded model answer per sample product, for the current prompt.

The sample's answers were recorded from one real model run. Replay, which means answering only
from stored files with no model call, serves them back. A cache key is a fingerprint of the model
name, prompts, schema, temperature and product text, so a change to any of those makes an entry miss.

sample/recording.json is the manifest of that run: the model, the date it was pinned, and a SHA-256
hash of each stored file. This check fails when a product has no stored answer, when a stored file
is an orphan that no product uses, when the manifest lacks a hash for a product, when a file differs
from its pinned hash (so an edited answer cannot pass as the recording), when a file does not have
the shape of a model answer, or when a replay of the sample misses. When every key is missing it
says in one line that the sample must be recorded again. It needs no key and no openai package.

    python3 tools/check_replay.py          check the stored answers against the manifest
    python3 tools/check_replay.py --pin    after recording again, write a new manifest

The recording is self-attested, which means the repo gives its word and no proof. The hashes show
the files have not changed since they were pinned, and they catch an accidental or partial edit.
They cannot prove the answers came from a model, since a person who writes an answer by hand and
pins it still passes.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cache import cache_key  # noqa: E402
from src.parsing import load_products  # noqa: E402
from src.pipeline import run_file  # noqa: E402
from src.prompts import SYSTEM_PROMPT, build_user_prompt, response_schema  # noqa: E402
from src.selector import DEFAULT_MODEL, SelectorConfig  # noqa: E402

CATALOG = ROOT / "sample" / "catalog.json"
CACHE_DIR = ROOT / "sample" / "cache"
MANIFEST = ROOT / "sample" / "recording.json"
ATTESTATION = ("Self-attested. Written by tools/check_replay.py --pin from the files then in sample/cache. The hashes show "
               "the files have not changed since. They do not prove the answers came from a model.")
RECORD = ("delete the files in sample/cache, run with a key: python3 run.py sample/catalog.json --cache-dir sample/cache "
          "--output output.json, then pin the new files: python3 tools/check_replay.py --pin")


def expected_keys() -> dict[str, str]:
    """Returns {cache key: product title} for every sample product, under the current prompt."""
    products, errors = load_products(json.loads(CATALOG.read_text()))
    if errors:
        raise SystemExit(f"malformed catalog records: {errors}")
    out = {}
    for p in products:
        user = build_user_prompt(p.product_title, p.product_description, p.search_results)
        out[cache_key(DEFAULT_MODEL, SYSTEM_PROMPT, user, response_schema(False), 0.0)] = p.product_title
    return out


def _is_count(value) -> bool:
    """True for a whole number above zero. A bool is not a count."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def shape_problem(stored) -> str | None:
    """Says why a stored file does not have the shape of a model answer, or returns None."""
    if not isinstance(stored, dict):
        return "is not a JSON object"
    ids = stored.get("trusted_result_ids")
    if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        return "has no list of whole-number result ids"
    if "source" in stored:
        return "is marked as written by hand"
    if not _is_count(stored.get("input_tokens")) or not _is_count(stored.get("output_tokens")):
        return "has no token counts"
    return None


def pin() -> int:
    """Writes sample/recording.json from the files now in sample/cache."""
    expected = expected_keys()
    files = {}
    for key in sorted(expected):
        path = CACHE_DIR / f"{key}.json"
        if not path.exists():
            print(f"FAIL: no stored answer for {expected[key]!r}, nothing was pinned. To record it, {RECORD}")
            return 1
        files[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"model": DEFAULT_MODEL, "pinned_on": datetime.date.today().isoformat(), "attestation": ATTESTATION, "files": files}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"pinned {len(files)} stored answers in sample/recording.json (self-attested, the hashes do not prove where the answers came from)")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--pin"]:
        return pin()
    expected = expected_keys()
    on_disk = {p.stem: p for p in CACHE_DIR.glob("*.json")} if CACHE_DIR.is_dir() else {}
    if not on_disk:
        print(f"FAIL: sample/cache is empty. To record it, {RECORD}")
        return 1

    problems = []
    missing = sorted(set(expected) - set(on_disk))
    orphans = sorted(set(on_disk) - set(expected))
    if missing and len(missing) == len(expected):
        problems.append(f"every key is missing. The prompt, schema, model or catalog changed, so the sample must be recorded again: {RECORD}")
    else:
        problems += [f"no stored answer for {expected[k]!r} (key {k[:12]})" for k in missing]
    problems += [f"orphan cache file {k[:12]} is used by no product" for k in orphans]
    try:
        manifest = json.loads(MANIFEST.read_text())
        pinned = manifest["files"]
        if manifest.get("model") != DEFAULT_MODEL or not isinstance(pinned, dict):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        pinned = None
        problems.append("sample/recording.json is missing, unreadable or names another model. After recording, write it with: python3 tools/check_replay.py --pin")
    if pinned is not None:
        problems += [f"sample/recording.json pins {k[:12]}, which no product uses" for k in sorted(set(pinned) - set(expected))]
        problems += [f"sample/recording.json has no hash for {expected[k]!r} (key {k[:12]})" for k in sorted(set(expected) - set(pinned))]
    for key in sorted(set(expected) & set(on_disk)):
        raw = on_disk[key].read_bytes()
        if pinned is not None and key in pinned and pinned[key] != hashlib.sha256(raw).hexdigest():
            problems.append(f"cache file {key[:12]} differs from the pinned recording in sample/recording.json")
        try:
            why = shape_problem(json.loads(raw))
        except ValueError:
            why = "is not valid JSON"
        if why:
            problems.append(f"cache file {key[:12]} {why}")

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
    print(f"OK: {hits} of {len(expected)} recorded answers replayed, 0 misses, 0 orphans, all match their pinned hash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
