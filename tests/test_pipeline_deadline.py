"""The wall-time budget must actually cut a run short, not just skip
starting new work while still blocking on everything already queued."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline as pipeline_module
from src.selector import ProductResult, SelectorConfig


class _SlowSelector:
    """Every call sleeps 0.3s then returns a trivial trusted result."""

    def __init__(self, config=None):
        pass

    def select(self, product):
        time.sleep(0.3)
        return ProductResult(key=product.key, product_title=product.product_title, llm_trusted_search_results=[])


def _write_input(tmp_path, n_products):
    data = {}
    for i in range(n_products):
        data[f"key{i}"] = {
            "product_title": f"Product {i}",
            "product_description": "",
            "search_results": {"1": "Title: A\nLink: https://a.com\nSnippet: b"},
            "trusted_search_results": [],
        }
    p = tmp_path / "in.json"
    p.write_text(json.dumps(data))
    return p


def test_deadline_stops_the_run_before_all_products_finish(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "Selector", _SlowSelector)
    input_path = _write_input(tmp_path, n_products=20)
    output_path = tmp_path / "out.json"

    start = time.monotonic()
    results, errors, deadline_hit = pipeline_module.run_file(
        input_path=input_path,
        output_path=output_path,
        config=SelectorConfig(),
        concurrency=1,  # serial, so 20 products * 0.3s = 6s if the deadline did nothing
        max_wall_seconds=0.5,
    )
    elapsed = time.monotonic() - start

    assert elapsed < 3.0, f"run took {elapsed:.2f}s, the wall-time budget did not actually bound it"
    assert len(results) < 20, "expected the deadline to leave some products unprocessed"
    assert deadline_hit is True, "a truncated run must report itself as incomplete, not a smaller success"


def test_no_deadline_processes_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "Selector", _SlowSelector)
    input_path = _write_input(tmp_path, n_products=3)
    output_path = tmp_path / "out.json"
    results, errors, deadline_hit = pipeline_module.run_file(
        input_path=input_path,
        output_path=output_path,
        config=SelectorConfig(),
        concurrency=3,
        max_wall_seconds=None,
    )
    assert len(results) == 3
    assert deadline_hit is False
