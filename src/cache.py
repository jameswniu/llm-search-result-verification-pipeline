"""A simple disk cache. It remembers the answer to a request so we never
ask the same exact question twice. Two reasons this matters here.

FIRST, THE ANSWER MUST NOT CHANGE WHEN YOU RUN IT AGAIN
  When you demo this, running it a second time
  has to show the exact same numbers, and it has to cost nothing. The cache "key" (a fingerprint of the request) is built from
  every single thing that could change the answer: the model name, the
  full instructions we send, the exact product being asked about, the
  shape we want the answer in, and the temperature setting. If any one
  of those changes, the fingerprint changes, and we correctly ask the
  question again instead of reusing a stale answer.

REPLAY MODE USES THE SAME CACHE READ-ONLY
  Replay means answering only from stored entries, with no model call. A
  cache opened with read_only=True never creates its folder and refuses to
  write. ReplayMiss is the error for an answer replay needed and did not
  find.

SECOND, TWO WORKERS CAN COLLIDE
  Several products get processed at the same time (see pipeline.py). Two
  of them could both try to save to the cache at the same moment. See
  the `set` method below for how we make that safe.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def cache_key(model: str, system_prompt: str, user_prompt: str, schema: dict, temperature: float) -> str:
    """Build one short fingerprint out of everything that can change the
    model's answer. Missing even one of these was a real bug we found
    during review: the fingerprint used to leave out temperature, so
    running at a different temperature could quietly reuse an answer
    that came from a different temperature. `sort_keys=True` makes sure
    the fingerprint comes out the same way every time, no matter what
    order Python happens to put the fields in."""
    payload = json.dumps(
        {"model": model, "system": system_prompt, "user": user_prompt, "schema": schema, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReplayMiss(Exception):
    """Raised when replay mode needs an answer that was never stored."""


class DiskCache:
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, read_only: bool = False):
        self.cache_dir = Path(cache_dir)
        self.read_only = read_only
        # A read-only cache is opened for replay. It never creates the
        # folder, so a typo in the path shows up as misses, not as a new
        # empty directory.
        if not read_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # If the saved file is broken or half-written, we just treat
            # it as if nothing was saved, and go ask the model again. We
            # never crash because of a bad cache file.
            return None

    def set(self, key: str, value: dict) -> None:
        if self.read_only:
            raise RuntimeError("cache is read-only")
        path = self._path(key)
        # We write to a temporary file first, with a random name, then
        # rename it into place. Why a RANDOM name and not a fixed one: if
        # two workers save to the SAME cache entry at the same time, and
        # they both used the same fixed temp filename, one worker could
        # delete the other's temp file before it finished, causing a
        # crash. A random name per save makes that collision impossible.
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(value))
        os.replace(tmp, path)  # renaming a file is instant and safe: nobody ever sees a half-written cache file
