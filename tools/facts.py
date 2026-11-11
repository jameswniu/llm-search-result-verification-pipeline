#!/usr/bin/env python3
"""The one place every number on the README and the figures comes from.

There are two groups, and every figure and sentence tags which one a number belongs to.

PRIVATE FILE numbers were measured on a private 50-product labeled file that is not distributed.
They are parsed from the referee cards in docs/REFEREE.md, so the page cannot disagree with the
cards. They are recorded and cannot be reproduced from this repo. A referee card is one table in
docs/REFEREE.md that states a number, how it was counted and what makes it look worse.

RECORDED SAMPLE numbers are recomputed by a replay run over sample/catalog.json. Replay serves
the answers in sample/cache without a model. Those answers were recorded from one real model run
and are not edited, so these numbers describe that one run on 8 invented products, not a benchmark.

    python3 tools/facts.py          print every number in a short table
    python3 tools/facts.py --short  the headline numbers on a few lines
    python3 tools/facts.py --json   the same as JSON

Needs no key and no openai package.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import ProductComparison, compute_metrics  # noqa: E402
from src.parsing import load_products  # noqa: E402
from src.pipeline import DEFAULT_CONCURRENCY, build_comparisons, run_file  # noqa: E402
from src.selector import MAX_RETRIES, SelectorConfig  # noqa: E402

REFEREE = ROOT / "docs" / "REFEREE.md"
CATALOG = ROOT / "sample" / "catalog.json"
CACHE_DIR = ROOT / "sample" / "cache"


def _find(pattern: str, text: str) -> tuple[str, ...]:
    """First match of a pattern, or exit. A number that cannot be read must stop the build."""
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"facts.py: cannot find in docs/REFEREE.md: {pattern}")
    return m.groups()


def private_numbers() -> dict:
    """Group (b): numbers parsed from the referee cards, recorded and not recomputed."""
    text = REFEREE.read_text()
    p = {}
    g = _find(r"\| What is claimed \| (0\.\d\d) to (0\.\d\d) precision, (0\.\d\d) recall, on the 10 held-out", text)
    p["held_precision_lo"], p["held_precision_hi"], p["held_recall"] = g
    g = _find(r"\| What is claimed \| (0\.\d\d) to (0\.\d\d) precision, (0\.\d\d) recall, over all 50", text)
    p["all_precision_lo"], p["all_precision_hi"], p["all_recall"] = g
    g = _find(r"\| What is claimed \| (0\.\d\d) precision with a 95% bootstrap interval of \[(0\.\d\d), (0\.\d\d)\], recall (0\.\d\d) to (0\.\d\d)", text)
    p["cv_precision"], p["cv_interval_lo"], p["cv_interval_hi"], p["cv_recall_lo"], p["cv_recall_hi"] = g
    g = _find(r"Keep every result: (0\.\d\d) precision, (1\.00) recall \((\d+) of the (\d+) results", text)
    p["keep_precision"], p["keep_recall"], p["keep_trusted"], p["keep_total"] = g
    g = _find(r"(\d+) recorded selection calls, ([\d,]+) input tokens and ([\d,]+) output tokens, which is a mean of ([\d,]+) input and ([\d.]+) output tokens per call", text)
    p["cost_calls"], p["cost_input_tokens"], p["cost_output_tokens"], p["cost_mean_input"], p["cost_mean_output"] = g
    g = _find(r"Input tokens at \$([\d.]+) per million and output tokens at \$([\d.]+) per million give \$(0\.\d+) in total, which is \$([\d.]+) per 100,000", text)
    p["price_input"], p["price_output"], p["cost_total"], p["cost_per_100k"] = g
    # the cost card must be arithmetic, not a typed number
    calls, tin, tout = int(p["cost_calls"]), int(p["cost_input_tokens"].replace(",", "")), int(p["cost_output_tokens"].replace(",", ""))
    total = tin / 1e6 * float(p["price_input"]) + tout / 1e6 * float(p["price_output"])
    if f"{total:.4f}" != p["cost_total"] or f"{total / calls * 100_000:.2f}" != p["cost_per_100k"]:
        raise SystemExit("facts.py: the cost card does not follow from its own token counts and prices")
    p["cost_rounded_per_100k"] = f"{round(float(p['cost_per_100k']))}"
    return p


def _constant(path: Path, name: str):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"facts.py: {name} not found in {path.name}")


def code_constants() -> dict:
    """Numbers read from the code, not typed: workers, retries, timeout and folds."""
    return {
        "workers": DEFAULT_CONCURRENCY,
        "retries": MAX_RETRIES,
        "timeout_s": int(SelectorConfig().request_timeout_s),
        "folds": _constant(ROOT / "cross_validate.py", "N_FOLDS"),
    }


def sample_numbers() -> dict:
    """Group (a): a replay run over the recorded sample, scored by src/metrics.py."""
    products, _ = load_products(json.loads(CATALOG.read_text()))
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
        results, _, _ = run_file(CATALOG, Path(tmp) / "out.json", SelectorConfig(cache_dir=CACHE_DIR, replay=True))
    report = compute_metrics(build_comparisons(results, products))
    keep_all = compute_metrics([
        ProductComparison(key=p.key, product_title=p.product_title,
                          predicted={int(r.result_id) for r in p.search_results}, truth=set(p.trusted_search_results))
        for p in products
    ])
    return {
        "n_products": report.n_products,
        "replayed": sum(1 for r in results if r.cache_hit and not r.error),
        "misses": sum(1 for r in results if r.error),
        "precision": f"{report.precision:.2f}",
        "recall": f"{report.recall:.2f}",
        "tp": report.total_true_positives,
        "fp": report.total_false_positives,
        "fn": report.total_false_negatives,
        "exact": report.n_exact_match,
        "dropped_ids": sum(len(r.dropped_ids) for r in results),
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "cost_usd": f"{sum(r.cost_usd for r in results):.4f}",
        "keep_precision": f"{keep_all.precision:.2f}",
        "keep_recall": f"{keep_all.recall:.2f}",
    }


def load() -> dict:
    return {"private": private_numbers(), "sample": sample_numbers(), "code": code_constants()}


def main() -> int:
    f = load()
    if sys.argv[1:] == ["--json"]:
        print(json.dumps(f, indent=2, sort_keys=True))
        return 0
    if sys.argv[1:] == ["--short"]:
        p, q = f["private"], f["sample"]
        print("PRIVATE FILE (recorded, cannot be reproduced from this repo)")
        print(f"  held-out 10   precision {p['held_precision_lo']} to {p['held_precision_hi']}, recall {p['held_recall']}")
        print(f"  pooled folds  precision {p['cv_precision']} [{p['cv_interval_lo']}, {p['cv_interval_hi']}], recall {p['cv_recall_lo']} to {p['cv_recall_hi']}")
        print(f"  keep all      precision {p['keep_precision']}, recall {p['keep_recall']}")
        print(f"  cost          about ${p['cost_rounded_per_100k']} per 100,000 products")
        print("RECORDED SAMPLE (one real model run, replayed with no key, not a benchmark)")
        print(f"  {q['replayed']} of {q['n_products']} replayed, precision {q['precision']}, recall {q['recall']}, {q['dropped_ids']} invented ids dropped")
        return 0
    for group, title in (("private", "PRIVATE FILE (recorded, parsed from docs/REFEREE.md, cannot be reproduced from this repo)"),
                         ("sample", "RECORDED SAMPLE (one real model run, replayed with no model call)"),
                         ("code", "CODE CONSTANTS (read from the code)")):
        print(f"== {title} ==")
        for k, v in f[group].items():
            print(f"  {k:<22} {v}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
