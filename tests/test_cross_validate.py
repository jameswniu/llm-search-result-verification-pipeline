"""Tests for cross_validate.py, the script that checks whether rules
written from part of the labeled data hold up on the part it never saw.
Covers the fold split, the bootstrap interval, the rule-stability table
and the retry behavior. Every model call is faked, so these tests need
no network and no API key."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cross_validate as cv
from src.cache import DiskCache
from src.metrics import ProductComparison
from src.parsing import parse_product
from src.selector import SelectorConfig


def _labeled_product(i, trusted, n_results=3):
    search_results = {str(r): f"Title: T{i}-{r}\nLink: https://x.com/{i}/{r}\nSnippet: s" for r in range(1, n_results + 1)}
    return parse_product(f"key{i}", {"product_title": f"Product {i}", "product_description": "", "search_results": search_results, "trusted_search_results": trusted})


def _make_50_products():
    products = []
    for i in range(38):
        products.append(_labeled_product(i, trusted=[1]))
    for i in range(38, 50):
        products.append(_labeled_product(i, trusted=[]))
    return products


def test_folds_partition_every_labeled_product_exactly_once():
    products = _make_50_products()
    folds = cv.make_folds(products, k=4)
    all_test_keys = [p.key for f in folds for p in f.test]
    assert sorted(all_test_keys) == sorted(p.key for p in products)
    assert len(set(all_test_keys)) == 50  # no duplicates, no product scored twice


def test_train_and_test_never_overlap_within_a_fold():
    products = _make_50_products()
    for f in cv.make_folds(products, k=4):
        assert {p.key for p in f.train}.isdisjoint({p.key for p in f.test})


def test_zero_trusted_products_spread_across_folds_not_clumped():
    products = _make_50_products()  # 12 products (indices 38-49) have an empty trusted list
    folds = cv.make_folds(products, k=4)
    zero_counts = [sum(1 for p in f.test if len(p.trusted_search_results) == 0) for f in folds]
    # 12 zero-trusted products over 4 folds should be 3 each, never e.g. [12, 0, 0, 0]
    assert zero_counts == [3, 3, 3, 3], zero_counts


def test_fold_split_is_deterministic_across_calls():
    products = _make_50_products()
    a = cv.make_folds(products, k=4)
    b = cv.make_folds(products, k=4)
    assert [sorted(p.key for p in f.test) for f in a] == [sorted(p.key for p in f.test) for f in b]


def test_bootstrap_ci_is_deterministic_and_brackets_the_point_estimate():
    comparisons = [ProductComparison(key=f"k{i}", product_title=f"p{i}", predicted={1}, truth={1}) for i in range(10)]
    comparisons += [ProductComparison(key="bad", product_title="bad", predicted={1, 2}, truth={1})]
    lo1, hi1 = cv.bootstrap_ci(comparisons, "precision", n_boot=200)
    lo2, hi2 = cv.bootstrap_ci(comparisons, "precision", n_boot=200)
    assert (lo1, hi1) == (lo2, hi2)  # fixed seed, reproducible
    assert 0.0 <= lo1 <= hi1 <= 1.0


def test_bootstrap_ci_empty_input_does_not_crash():
    assert cv.bootstrap_ci([], "precision") == (0.0, 0.0)


class _FakeSynthesisAndScoringClient:
    """Fakes both the synthesis call (returns a fixed rules_block) and the
    scoring call (trusts everything), so run_cross_validation can be
    exercised end to end with no network access."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        schema_name = kwargs["response_format"]["json_schema"]["name"]
        if schema_name == "synthesis":
            content = {"rules_block": "1. A fake synthesized rule."}
        elif schema_name == "selection":
            content = {"trusted_result_ids": [1]}
        else:
            raise AssertionError(f"unexpected schema {schema_name}")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def test_run_cross_validation_scores_every_product_exactly_once(tmp_path):
    products = _make_50_products()
    client = _FakeSynthesisAndScoringClient()
    cache = DiskCache(cache_dir=tmp_path / ".cache")
    result = cv.run_cross_validation(products, SelectorConfig(use_cache=False), cache=cache, client=client)
    assert len(result.pooled_comparisons) == 50
    assert len(result.folds) == 4
    assert client.calls == 54  # 4 synthesis calls + 50 scoring calls


def test_run_shipped_prompt_variance_does_not_call_synthesis(tmp_path):
    products = _make_50_products()
    client = _FakeSynthesisAndScoringClient()
    cache = DiskCache(cache_dir=tmp_path / ".cache")
    result = cv.run_shipped_prompt_variance(products, SelectorConfig(use_cache=False), cache=cache, client=client)
    assert len(result.pooled_comparisons) == 50
    assert client.calls == 50  # scoring only, no synthesis calls


def test_stability_table_flags_disagreeing_stances_as_conflicting_not_stable():
    # This is the exact defect a code review caught: 4 folds each
    # mentioning "size" but split on ignore vs require must never render
    # as "stable (all folds)", since that would report contradictory rules
    # as independent corroboration.
    concepts = [{"concept": "size handling", "mentions": [{"fold": 0, "stance": "require"}, {"fold": 1, "stance": "ignore"}, {"fold": 2, "stance": "ignore"}, {"fold": 3, "stance": "ignore"}]}]
    table = cv.format_stability_table(concepts)
    assert "CONFLICTING" in table
    assert "stable, all folds agree" not in table


def test_stability_table_marks_genuine_unanimous_agreement_as_stable():
    concepts = [{"concept": "generic items excluded", "mentions": [{"fold": i, "stance": "ignore"} for i in range(4)]}]
    table = cv.format_stability_table(concepts)
    assert "stable, all folds agree" in table


def test_stability_table_marks_single_fold_concept_as_likely_artifact():
    concepts = [{"concept": "a very specific one-off rule", "mentions": [{"fold": 3, "stance": "require"}]}]
    table = cv.format_stability_table(concepts)
    assert "single-fold" in table


class _FlakyThenOkClient:
    """Fails with a transient connection error the first N calls, then
    succeeds, for exercising the retry path on the synthesis call."""

    def __init__(self, n_failures, content):
        from openai import APIConnectionError

        self.chat = SimpleNamespace(completions=self)
        self.n_failures = n_failures
        self.content = content
        self.calls = 0
        self._error_cls = APIConnectionError

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.n_failures:
            raise self._error_cls(message="down", request=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.content)))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def test_synthesis_retries_on_transient_error_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(cv.time, "sleep", lambda _s: None)
    products = _make_50_products()[:5]
    client = _FlakyThenOkClient(n_failures=1, content={"rules_block": "1. A rule."})
    cache = DiskCache(cache_dir=tmp_path / ".cache")
    synth = cv.RuleSynthesizer(client=client, cache=cache, use_cache=False)
    rules_block, usage = synth.synthesize(products)
    assert rules_block == "1. A rule."
    assert client.calls == 2  # one failure, then a success


def test_synthesis_raises_a_clean_error_after_exhausting_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(cv.time, "sleep", lambda _s: None)
    products = _make_50_products()[:5]
    client = _FlakyThenOkClient(n_failures=99, content={"rules_block": "unused"})
    cache = DiskCache(cache_dir=tmp_path / ".cache")
    synth = cv.RuleSynthesizer(client=client, cache=cache, use_cache=False)
    try:
        synth.synthesize(products)
        assert False, "expected SynthesisFailed"
    except cv.SynthesisFailed:
        pass


def test_no_cache_flag_actually_bypasses_the_synthesis_cache(tmp_path):
    products = _make_50_products()[:5]
    client = _FakeSynthesisAndScoringClient()
    cache = DiskCache(cache_dir=tmp_path / ".cache")
    synth = cv.RuleSynthesizer(client=client, cache=cache, use_cache=False)
    synth.synthesize(products)
    synth.synthesize(products)  # same input, use_cache=False must call the API again
    assert client.calls == 2
