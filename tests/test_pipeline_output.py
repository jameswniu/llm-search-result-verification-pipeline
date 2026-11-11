"""Guards the output contract: llm_trusted_search_results is
written exactly as asked, and the optional reason and confidence fields
stay off unless they are turned on."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import _write_output
from src.selector import ProductResult


def test_default_output_adds_only_the_promised_key(tmp_path):
    original = {
        "some key": {
            "product_title": "T",
            "product_description": "",
            "search_results": {"1": "Title: A\nLink: https://a.com\nSnippet: b"},
            "trusted_search_results": [1],
        }
    }
    results = {"some key": ProductResult(key="some key", product_title="T", llm_trusted_search_results=[1])}
    out_path = tmp_path / "out.json"
    _write_output(out_path, original, results)
    written = json.loads(out_path.read_text())

    new_keys = set(written["some key"].keys()) - set(original["some key"].keys())
    assert new_keys == {"llm_trusted_search_results"}, (
        f"output must add exactly llm_trusted_search_results and nothing else by default, got {new_keys}"
    )
    assert written["some key"]["llm_trusted_search_results"] == [1]


def test_a_malformed_original_record_does_not_crash_the_checkpoint(tmp_path):
    # load_products() already rejected this record (not a dict), but the
    # original file still has to round-trip through every checkpoint write
    # without dict()-coercing something that was never a dict.
    original = {
        "good key": {
            "product_title": "T",
            "product_description": "",
            "search_results": {},
            "trusted_search_results": [],
        },
        "bad key": None,
    }
    results = {"good key": ProductResult(key="good key", product_title="T", llm_trusted_search_results=[])}
    out_path = tmp_path / "out.json"
    _write_output(out_path, original, results)  # must not raise
    written = json.loads(out_path.read_text())
    assert written["bad key"] is None
    assert written["good key"]["llm_trusted_search_results"] == []


def test_explanation_fields_only_appear_when_the_result_carries_them(tmp_path):
    original = {"k": {"product_title": "T", "product_description": "", "search_results": {}, "trusted_search_results": []}}
    results = {"k": ProductResult(key="k", product_title="T", llm_trusted_search_results=[], reason="matched", confidence=0.9)}
    out_path = tmp_path / "out.json"
    _write_output(out_path, original, results)
    written = json.loads(out_path.read_text())
    assert written["k"]["llm_reason"] == "matched"
    assert written["k"]["llm_confidence"] == 0.9
