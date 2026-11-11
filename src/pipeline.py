"""Runs selector.py's "ask about one product" logic across a WHOLE file
of products instead of just one. selector.py answers "is asking about
one product safe and correctly priced". This file answers "can we ask
about 50 (or 100,000) products, unattended, in a reasonable amount of
time, without losing our work if something crashes partway through".

THE WHOLE RUN IN FOUR PARTS
  1. Read the file, and skip (don't crash on) any row that's broken.
  2. Hand every product to a small pool of workers that ask about
     several products at the same time.
  3. The moment each answer comes back, immediately save the output file
     to disk. That way, if the program gets killed, crashes, or runs out
     of time, we never lose the answers we already got.
  4. If we run out of time before finishing everyone, say so honestly
     (deadline_hit=True) instead of pretending a partly-finished file is
     a complete success.

REPLAY AND THE RUN SUMMARY
  In replay mode answers come only from the cache and nothing is called.
  print_run_summary then reports hits and misses, says plainly when the
  answers were scripted by hand, and prints no dollar cost for them.

WHY WORKERS RUNNING AT THE SAME TIME, INSTEAD OF ONE AT A TIME
  Each question to the model takes about a second and just sits there
  waiting for OpenAI to answer. If we asked one at a time, 50 products
  would take 50 seconds just waiting around doing nothing useful. Asking
  several at once means we're mostly just waiting in parallel instead of
  one after another. DEFAULT_CONCURRENCY caps this at 8 at a time: enough
  to get through 50 products in well under a minute, but not so many at
  once that OpenAI complains about too many requests.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .cache import DEFAULT_CACHE_DIR
from .metrics import ProductComparison, compute_metrics, format_report
from .parsing import Product, load_products
from .selector import ProductResult, Selector, SelectorConfig

DEFAULT_CONCURRENCY = 8
COST_PROJECTION_N = 100_000  # an illustrative production size


def run_file(
    input_path: Path,
    output_path: Path,
    config: SelectorConfig,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_wall_seconds: float | None = None,
    limit: int | None = None,
) -> tuple[list[ProductResult], list[str], bool]:
    """Returns (results, parse_errors, deadline_hit). Saves the output
    file after every single product finishes, and once more at the very
    end. deadline_hit is True when we ran out of time before finishing
    everyone, which means the output file is NOT complete, and whoever
    calls this function needs to treat that as a failure, not as "a
    smaller success"."""
    data = json.loads(Path(input_path).read_text())
    products, parse_errors = load_products(data)
    if limit is not None:
        products = products[:limit]  # a quick test run using just the first N products, always the SAME first N, never a random pick, so reruns are comparable

    for e in parse_errors:
        print(f"[skip] malformed record: {e}", file=sys.stderr)

    # Print a line right at the start, not just at the end. If this gets
    # launched live, anyone watching should be able to see right away
    # that it's actually running, on the right file, with the settings
    # it was given.
    print(
        f"[start] {len(products)} products from {input_path} "
        f"(model={config.model}, concurrency={concurrency}, cache={_cache_label(config)})",
        file=sys.stderr,
    )
    selector = Selector(config=config)
    results: dict[str, ProductResult] = {}
    start = time.monotonic()
    deadline_hit = False

    def _checkpoint():
        _write_output(output_path, data, results)

    # ------------------------------------------------------------------
    # STEP 1. Hand every product to the worker pool right away. The pool's
    # own size limit (concurrency), not this loop, is what actually
    # controls how many run at once.
    # ------------------------------------------------------------------
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = {pool.submit(selector.select, p): p for p in products}
        pending = set(futures)

        # ------------------------------------------------------------------
        # STEP 2. Handle answers one at a time as they come in, saving
        # after each one, while never waiting past our time budget for
        # the next answer.
        # ------------------------------------------------------------------
        while pending:
            if max_wall_seconds is not None:
                remaining = max_wall_seconds - (time.monotonic() - start)
                if remaining <= 0:
                    deadline_hit = True
                    break
            else:
                remaining = None

            # This line is what actually enforces the time limit while
            # the run is happening. An earlier version of this code only
            # checked the time limit BEFORE starting new questions, and
            # then just waited forever for whatever was already running,
            # which for a small batch is basically no time limit at all.
            # This version can stop between ANY two answers, not just
            # before we start asking new questions.
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            if not done:  # we hit our own time limit and nothing new finished in that window
                deadline_hit = True
                break
            for future in done:
                p = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # a bug in OUR code, not a problem with OpenAI; we still don't want this to crash the whole run
                    result = ProductResult(key=p.key, product_title=p.product_title, llm_trusted_search_results=[], error=f"internal error: {exc}")
                results[p.key] = result
                _checkpoint()  # saved after EVERY single answer, not in batches, so a crash mid-run never loses anything that already finished
                status = "error" if result.error else ("cached" if result.cache_hit else f"{len(result.llm_trusted_search_results)} trusted")
                print(f"[done] {len(results)}/{len(products)} {p.product_title!r}: {status}", file=sys.stderr)

        if deadline_hit:
            for future in pending:
                future.cancel()  # this only actually stops a question that hasn't started yet; one already in progress keeps running
            for p in products:
                if p.key not in results:
                    print(f"[deadline] wall-time budget hit before finishing {p.product_title!r}", file=sys.stderr)
    finally:
        # wait=False means: a time limit is a time limit. If a question
        # was already in progress when time ran out, we let it keep
        # running quietly in the background instead of making the whole
        # program wait around for it to finish.
        #
        # KNOWN LIMIT: "in progress" can still mean up to MAX_RETRIES vendor
        # retries with backoff (see selector.py), so an in-flight product can
        # keep spending API calls for tens of seconds past max_wall_seconds.
        # The deadline bounds when the CLI STOPS STARTING new work and when
        # it reports back to the caller, not the last byte of spend. Fixing
        # that needs an absolute deadline threaded into Selector.select()
        # itself; flagged, not fixed here.
        pool.shutdown(wait=not deadline_hit, cancel_futures=deadline_hit)

    _checkpoint()
    print(f"[written] {output_path} ({len(results)}/{len(products)} products, {time.monotonic() - start:.1f}s)", file=sys.stderr)
    results_list = [results[p.key] for p in products if p.key in results]
    return results_list, parse_errors, deadline_hit


def _write_output(output_path: Path, original_data: dict, results: dict[str, ProductResult]) -> None:
    """Rewrites the WHOLE output file, every time, using the original
    input data plus whatever answers we have so far. This gets called
    after every single product finishes, so it has to be fast and safe
    to run while the rest of the program is still going. That's why we
    write to a temporary file first and then swap it in, instead of
    editing the real file bit by bit."""
    out = {}
    for key, value in original_data.items():
        # If load_products() already decided this row was too broken to
        # use (it's not even a proper dictionary), we just copy it
        # through unchanged instead of trying to work with it. Trying to
        # process it here would crash, and that would take down the
        # whole save, including every good answer we already have.
        if not isinstance(value, dict):
            out[key] = value
            continue
        record = dict(value)
        if key in results:
            r = results[key]
            record["llm_trusted_search_results"] = r.llm_trusted_search_results  # the one field in the output contract; always present once this product is scored
            if r.reason is not None:
                record["llm_reason"] = r.reason
            if r.confidence is not None:
                record["llm_confidence"] = r.confidence
            if r.error:
                record["llm_error"] = r.error
        out[key] = record
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(output_path)  # swapping in a finished file is instant, so nobody reading the output file ever sees a half-written one


def _cache_label(config: SelectorConfig) -> str:
    if config.replay:
        return f"replay:{config.cache_dir if config.cache_dir is not None else DEFAULT_CACHE_DIR}"
    return "on" if config.use_cache else "off"


def print_run_summary(results: list[ProductResult], replay: bool = False) -> None:
    n = len(results)
    errors = [r for r in results if r.error]
    cache_hits = sum(1 for r in results if r.cache_hit)
    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)
    total_cost = sum(r.cost_usd for r in results)
    latencies = [r.latency_s for r in results if not r.cache_hit]  # a reused (cached) answer took basically no time, and would make the real average look artificially fast
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    priced_calls = n - cache_hits
    avg_cost_per_product = total_cost / priced_calls if priced_calls else 0.0
    projected_100k = avg_cost_per_product * COST_PROJECTION_N  # scaling up to an illustrative production size

    if replay:
        misses = sum(1 for r in results if r.error and r.error.startswith("ReplayMiss"))
        print(f"== Run summary, replayed ({n} products, {len(errors)} errors) ==")
        print(f"Replay: {cache_hits} hits, {misses} misses")
        print("Latency: n/a (replayed)")
        scripted = sum(1 for r in results if r.scripted)
        if scripted:
            print(f"Answers: {scripted} scripted by hand, no model was called, so there is no cost")
            dropped = sum(len(r.dropped_ids) for r in results)
            print(f"Dropped ids (not among a product's results): {dropped}")
        else:
            print(f"Tokens: {total_input} input, {total_output} output (stored)")
            print(f"Cost of the stored calls: ${total_cost:.4f}  (from stored token counts)")
    else:
        print(f"== Run summary ({n} products, {len(errors)} errors, {cache_hits} cache hits) ==")
        print(f"Avg latency per (non-cached) call: {avg_latency:.2f}s")
        print(f"Tokens: {total_input} input, {total_output} output")
        print(f"Cost this run: ${total_cost:.4f}  (avg ${avg_cost_per_product:.5f}/product)")
        print(f"Projected cost for {COST_PROJECTION_N:,} products: ${projected_100k:,.2f}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for r in errors:
            print(f"  {r.product_title!r}: {r.error}")


def build_comparisons(results: list[ProductResult], products: list[Product]) -> list[ProductComparison]:
    """Turns our run's answers into the (what we guessed, what was
    correct) pairs that metrics.py actually scores. This logic lives
    here, not in metrics.py, because this is the one place that knows
    what a Product and a ProductResult look like. metrics.py never
    needs to know that."""
    products_by_key = {p.key: p for p in products}
    comparisons = []
    for r in results:
        p = products_by_key.get(r.key)
        if p is None or p.trusted_search_results is None:
            # This product has no correct answer to check against (a
            # fresh, unlabeled test file has none at all), so there's
            # nothing to compare. We skip it instead of pretending it has
            # an empty correct answer.
            continue
        comparisons.append(
            ProductComparison(
                key=r.key,
                product_title=r.product_title,
                predicted=set(r.llm_trusted_search_results),
                truth=set(p.trusted_search_results),
            )
        )
    return comparisons
