"""Tests for replay mode: answers come only from a stored cache, a miss is
a loud per-product error, and nothing needs a key or the openai package.
Replay means answering only from stored entries with no model call. _NoClient is
the stand-in client that raises if replay ever tries to call the model."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import prompts
from src.cache import DiskCache, ReplayMiss, cache_key
from src.parsing import parse_product
from src.prompts import SYSTEM_PROMPT, build_user_prompt, response_schema
from src.selector import DEFAULT_MODEL, Selector, SelectorConfig, _NoClient



def _product():
    results = {str(i): f"Title: T{i}\nLink: https://x.example.com/{i}\nSnippet: s{i}" for i in range(1, 4)}
    return parse_product("key", {"product_title": "Test Product", "product_description": "", "search_results": results})


def _key(product, system_prompt=None):
    user = build_user_prompt(product.product_title, product.product_description, product.search_results)
    return cache_key(DEFAULT_MODEL, system_prompt or SYSTEM_PROMPT, user, response_schema(False), 0.0)


def _seed(cache_dir, product):
    DiskCache(cache_dir).set(_key(product), {"trusted_result_ids": [1, 3], "reason": "r", "confidence": 0.9, "input_tokens": 1000, "output_tokens": 100})


def test_replay_hit_returns_the_stored_answer_and_never_touches_the_client(tmp_path):
    product = _product()
    _seed(tmp_path / "c", product)
    selector = Selector(config=SelectorConfig(cache_dir=tmp_path / "c", replay=True))
    assert isinstance(selector.client, _NoClient)
    result = selector.select(product)
    assert result.error is None
    assert result.llm_trusted_search_results == [1, 3]
    assert result.cache_hit is True
    assert result.cost_usd > 0  # priced from the stored token counts, not zero


def test_replay_miss_is_an_error_makes_no_call_and_writes_nothing(tmp_path):
    cache_dir = tmp_path / "empty"
    cache_dir.mkdir()
    selector = Selector(config=SelectorConfig(cache_dir=cache_dir, replay=True))
    result = selector.select(_product())
    assert "ReplayMiss" in result.error
    assert result.llm_trusted_search_results == []
    assert list(cache_dir.iterdir()) == []


def test_read_only_cache_does_not_create_the_folder_and_refuses_writes(tmp_path):
    missing = tmp_path / "nope"
    cache = DiskCache(missing, read_only=True)
    assert not missing.exists()
    assert cache.get("abc") is None
    with pytest.raises(RuntimeError, match="read-only"):
        cache.set("abc", {})


def test_changed_prompt_turns_every_hit_into_a_miss(tmp_path, monkeypatch):
    product = _product()
    _seed(tmp_path / "c", product)
    monkeypatch.setattr("src.selector.SYSTEM_PROMPT", SYSTEM_PROMPT + " changed")
    selector = Selector(config=SelectorConfig(cache_dir=tmp_path / "c", replay=True))
    result = selector.select(product)
    assert "ReplayMiss" in result.error
    assert "changed since the cache was built" in result.error


def test_replay_works_with_the_openai_package_blocked(tmp_path):
    product = _product()
    _seed(tmp_path / "c", product)
    code = (
        "import sys; sys.modules['openai'] = None\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from src.selector import Selector, SelectorConfig, OpenAI\n"
        "from tests.test_replay import _product\n"
        "assert OpenAI is None\n"
        f"s = Selector(config=SelectorConfig(cache_dir={str(tmp_path / 'c')!r}, replay=True))\n"
        "r = s.select(_product())\n"
        "assert r.error is None and r.llm_trusted_search_results == [1, 3], r\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_non_replay_without_openai_package_exits_with_an_install_hint(tmp_path, monkeypatch):
    monkeypatch.setattr("src.selector.OpenAI", None)
    with pytest.raises(SystemExit, match="pip install"):
        Selector(config=SelectorConfig(cache_dir=tmp_path / "c"))


def test_sample_replays_end_to_end_with_no_key(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    out = tmp_path / "out.json"
    done = subprocess.run(
        [sys.executable, "run.py", "sample/catalog.json", "--cache-dir", "sample/cache", "--replay", "--output", str(out)],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )
    assert done.returncode == 0, done.stderr
    assert "0 errors" in done.stdout
    assert "scripted by hand, no model was called" in done.stdout
    assert "$" not in done.stdout.split("== Scripted")[0]  # no dollar cost is printed for scripted answers
    written = json.loads(out.read_text())
    # imperfection 1: the made-up id 9 is dropped, and the summary says so
    assert written["Harbor Pilsener Lager 6 pk"]["llm_trusted_search_results"] == [1, 2]
    assert "Dropped ids (not among a product's results): 1" in done.stdout
    # imperfection 2: one near miss is kept, so precision is below 1.00
    assert written["FizzWell Cola Zero Sugar 12 fl oz can"]["llm_trusted_search_results"] == [1, 2, 5, 6]
    assert "Precision:            0.947" in done.stdout


def test_sample_cache_matches_the_scripted_answers():
    done = subprocess.run([sys.executable, "tools/check_replay.py"], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stdout


def test_replay_is_refused_with_no_cache_or_with_explanation(tmp_path):
    for flag in ("--no-cache", "--with-explanation"):
        done = subprocess.run(
            [sys.executable, "run.py", "sample/catalog.json", "--replay", flag, "--output", str(tmp_path / "o.json")],
            capture_output=True, text=True, cwd=ROOT,
        )
        assert done.returncode == 2
        assert "--replay cannot be combined" in done.stderr


def test_output_pointing_at_the_input_file_is_refused_and_leaves_it_untouched(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text((ROOT / "sample" / "catalog.json").read_text())
    before = catalog.read_text()
    done = subprocess.run(
        [sys.executable, "run.py", str(catalog), "--cache-dir", "sample/cache", "--replay", "--output", str(catalog)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert done.returncode == 2
    assert "same file as the input" in done.stderr
    assert str(catalog) in done.stderr
    assert catalog.read_text() == before
