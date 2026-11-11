"""Tests for src/selector.py, which asks the model which search results to
trust for one product. Covers dropping made-up ids, skipping the call when
there are no results, cache hits, retries, and the optional reason and
confidence fields. The API client is faked and the cache is a temporary
folder, so no real call is made."""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache import DiskCache
from src.parsing import parse_product
from src.selector import Selector, SelectorConfig


def _product(trusted_ids=None, n_results=3):
    search_results = {str(i): f"Title: T{i}\nLink: https://x.com/{i}\nSnippet: s{i}" for i in range(1, n_results + 1)}
    return parse_product(
        "key",
        {
            "product_title": "Test Product",
            "product_description": "",
            "search_results": search_results,
            "trusted_search_results": trusted_ids or [],
        },
    )


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        content_json, usage = item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content_json)))],
            usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]),
        )


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def _cache(tmp_path):
    return DiskCache(cache_dir=tmp_path / ".cache")


def test_drops_result_ids_outside_the_valid_set(tmp_path):
    # A model that hallucinates an id (e.g. 99 on a 3-result product) must
    # not have that id survive into llm_trusted_search_results.
    product = _product(n_results=3)
    client = FakeClient([({"trusted_result_ids": [1, 99]}, (10, 5))])
    selector = Selector(config=SelectorConfig(use_cache=False), client=client, cache=_cache(tmp_path))
    result = selector.select(product)
    assert result.llm_trusted_search_results == [1]
    assert result.error is None


def test_empty_search_results_short_circuits_with_no_api_call(tmp_path):
    product = _product(n_results=0)
    client = FakeClient([])  # would raise IndexError if called
    selector = Selector(config=SelectorConfig(use_cache=False), client=client, cache=_cache(tmp_path))
    result = selector.select(product)
    assert result.llm_trusted_search_results == []
    assert client.chat.completions.calls == 0


def test_cache_hit_skips_the_second_api_call(tmp_path):
    product = _product(n_results=2)
    client = FakeClient([({"trusted_result_ids": [1]}, (10, 5))])
    cache = _cache(tmp_path)
    cfg = SelectorConfig(use_cache=True)
    s1 = Selector(config=cfg, client=client, cache=cache)
    r1 = s1.select(product)
    assert r1.cache_hit is False
    assert client.chat.completions.calls == 1

    s2 = Selector(config=cfg, client=client, cache=cache)  # same client, response list now empty
    r2 = s2.select(product)
    assert r2.cache_hit is True
    assert r2.llm_trusted_search_results == [1]
    assert client.chat.completions.calls == 1  # unchanged: second call never hit the API


def test_transient_error_retries_then_succeeds(tmp_path):
    from openai import APIConnectionError

    product = _product(n_results=2)
    transient = APIConnectionError(message="transient", request=None)
    client = FakeClient([transient, ({"trusted_result_ids": [1]}, (10, 5))])
    selector = Selector(config=SelectorConfig(use_cache=False, max_retries=2), client=client, cache=_cache(tmp_path))
    # avoid a real sleep in the test
    import src.selector as selector_module

    selector_module.time.sleep = lambda _s: None
    result = selector.select(product)
    assert result.llm_trusted_search_results == [1]
    assert client.chat.completions.calls == 2


def test_exhausted_retries_return_empty_with_error_not_a_crash(tmp_path):
    from openai import APIConnectionError

    product = _product(n_results=2)
    client = FakeClient([APIConnectionError(message="down", request=None)] * 3)
    selector = Selector(config=SelectorConfig(use_cache=False, max_retries=2), client=client, cache=_cache(tmp_path))
    import src.selector as selector_module

    selector_module.time.sleep = lambda _s: None
    result = selector.select(product)
    assert result.llm_trusted_search_results == []
    assert result.error is not None


def test_explanation_fields_absent_unless_flag_is_on(tmp_path):
    product = _product(n_results=1)
    client = FakeClient([({"trusted_result_ids": [1], "reason": "matches", "confidence": 0.9}, (10, 5))])
    selector = Selector(config=SelectorConfig(use_cache=False, include_explanation=False), client=client, cache=_cache(tmp_path))
    result = selector.select(product)
    assert result.reason is None
    assert result.confidence is None
