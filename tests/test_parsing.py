"""Tests for src/parsing.py, which turns the raw products JSON into typed
records. Covers a normal search-result text, a malformed one that must
not crash, the trap where a product's dict key differs from its inner
title, and bad records that get skipped and reported, not fatal."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsing import load_products, parse_product, parse_search_result


def test_parses_a_normal_result_blob():
    r = parse_search_result("1", "Title: Foo Bar\nLink: https://example.com/x\nSnippet: some text here")
    assert r.title == "Foo Bar"
    assert r.link == "https://example.com/x"
    assert r.snippet == "some text here"


def test_malformed_blob_does_not_raise():
    r = parse_search_result("2", "not a valid blob at all")
    assert r.result_id == "2"
    assert r.title == ""
    assert r.snippet == "not a valid blob at all"


def test_keys_off_inner_fields_not_the_dict_key():
    # This is the concatenated-key trap in the input file: the dict
    # key smashes title+description together, but product_title inside the
    # value is clean. Parsing must read the inner field, never the key.
    key = "protein water water 4 ea water, protein, punch"
    value = {
        "product_title": "protein water 4 ea",
        "product_description": "water, protein, punch",
        "search_results": {"1": "Title: X\nLink: https://a.com\nSnippet: y"},
        "trusted_search_results": [],
    }
    product = parse_product(key, value)
    assert product.product_title == "protein water 4 ea"
    assert product.product_description == "water, protein, punch"
    assert product.key == key  # kept only for round-tripping output


def test_trailing_space_keys_do_not_affect_parsed_fields():
    key = "Lipton CB Unswet Blk Tea 14oz "
    value = {
        "product_title": "Lipton CB Unswet Blk Tea 14oz",
        "product_description": "",
        "search_results": {"1": "Title: X\nLink: https://a.com\nSnippet: y"},
        "trusted_search_results": [],
    }
    product = parse_product(key, value)
    assert product.product_title == "Lipton CB Unswet Blk Tea 14oz"


def test_missing_product_title_raises_value_error():
    try:
        parse_product("bad", {"search_results": {}})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_load_products_skips_malformed_and_reports_error():
    data = {
        "good key": {
            "product_title": "Good Product",
            "product_description": "",
            "search_results": {"1": "Title: A\nLink: https://a.com\nSnippet: b"},
            "trusted_search_results": [1],
        },
        "bad key": {"product_description": "no title, no search_results"},
    }
    products, errors = load_products(data)
    assert len(products) == 1
    assert products[0].product_title == "Good Product"
    assert len(errors) == 1
    assert "bad key" in errors[0]


def test_search_results_preserve_numeric_order():
    value = {
        "product_title": "T",
        "product_description": "",
        "search_results": {
            "10": "Title: J\nLink: https://j.com\nSnippet: j",
            "2": "Title: B\nLink: https://b.com\nSnippet: b",
            "1": "Title: A\nLink: https://a.com\nSnippet: a",
        },
        "trusted_search_results": [],
    }
    product = parse_product("k", value)
    assert [r.result_id for r in product.search_results] == ["1", "2", "10"]
