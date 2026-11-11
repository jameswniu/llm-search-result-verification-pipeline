"""Tests for src/cache.py: the cache key must change when the temperature changes and stay the same
for identical inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache import cache_key


def test_different_temperature_gives_a_different_cache_key():
    # Guards a real bug caught in review: the key used to omit temperature,
    # so a rerun at a different temperature would silently reuse a cached
    # answer generated under a different sampling setting.
    k1 = cache_key("gpt-4o-mini", "sys", "user", {"a": 1}, temperature=0.0)
    k2 = cache_key("gpt-4o-mini", "sys", "user", {"a": 1}, temperature=0.9)
    assert k1 != k2


def test_same_inputs_give_the_same_key():
    k1 = cache_key("gpt-4o-mini", "sys", "user", {"a": 1}, temperature=0.0)
    k2 = cache_key("gpt-4o-mini", "sys", "user", {"a": 1}, temperature=0.0)
    assert k1 == k2
