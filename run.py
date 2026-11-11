#!/usr/bin/env python3
"""The command-line program. This file is just glue: read the flags the
user typed, wire together parsing.py, pipeline.py, selector.py, and
metrics.py, print the result, and pick an exit code. All the real work
happens in the src/ folder; this file is just what a person actually
types and runs.

WHY THE API KEY ONLY EVER COMES FROM THE ENVIRONMENT
  We read it from the environment only. Export it in your shell before
  you run this. We never accept the key as a typed-in
  command-line option: anything typed on the command line gets saved in
  your shell history and can be seen by other programs on the machine
  while it's running. An environment variable is much harder to leak by
  accident.

WHAT --replay DOES
  It serves answers only from the cache folder given by --cache-dir, with
  no key, no openai package and no network. A product with no stored
  answer is an error and the run exits 1. The sample in sample/ uses this
  with hand-written answers.

WHY THE EXIT CODE MATTERS
  When this program finishes, it reports back a single number: 0 for
  "everything's fine", or a non-zero number for "something's wrong". A
  grading script, a CI system, or a person just checking the last exit
  code all rely on that number. This program only returns 0 when the
  output file can actually be trusted, meaning no broken input rows, no
  product that failed to get an answer, and no run that got cut off
  early because it ran out of time. A run that finished cleanly but
  happened to disagree with a lot of the correct answers still returns 0,
  disagreeing with the answers is not the same thing as the PROGRAM
  failing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import build_comparisons, print_run_summary, run_file
from src.parsing import load_products
from src.metrics import compute_metrics, format_report
from src.selector import SelectorConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Select trusted search results for products.")
    parser.add_argument("input", type=Path, help="Path to a products JSON file (see sample/catalog.json for the shape).")
    parser.add_argument("--output", type=Path, default=Path("output.json"), help="Where to write llm_trusted_search_results.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent product calls.")
    parser.add_argument("--max-wall-seconds", type=float, default=None, help="Stop starting new products past this wall-clock budget.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N products (for a quick smoke run).")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the disk cache (forces a live call every time).")
    parser.add_argument("--with-explanation", action="store_true", help="Also write reason and confidence for each product. Off by default, since the output contract is llm_trusted_search_results only.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Where the disk cache lives (default: .cache in the repo).")
    parser.add_argument("--replay", action="store_true", help="Serve answers only from the cache. No API key, no openai package, no network. A product with no stored answer is an error and the run exits 1.")
    parser.add_argument("--held-out-only", action="store_true", help="Only run the held-out split (see DECISIONS.md) and report metrics on it.")
    args = parser.parse_args()

    # Check for the most likely live mistake FIRST, before we even touch
    # the API key or the cache: pointing this at a file path that doesn't
    # actually exist, because the products file got saved somewhere
    # unexpected.
    if not args.input.is_file():
        print(f"Input file not found: {args.input}  (try `python3 run.py sample/catalog.json --replay --cache-dir sample/cache`)", file=sys.stderr)
        return 2

    if args.output.resolve() == args.input.resolve():
        print(f"--output {args.output} is the same file as the input {args.input}: the run would overwrite the input. Choose a different --output.", file=sys.stderr)
        return 2

    if args.replay and (args.no_cache or args.with_explanation):
        print("--replay cannot be combined with --no-cache or --with-explanation: replay only reads stored answers, and the explanation schema was never stored.", file=sys.stderr)
        return 2

    if not args.replay and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Export it first (`export OPENAI_API_KEY=...`) or use `--replay` with a stored cache.", file=sys.stderr)
        return 2

    config = SelectorConfig(
        model=args.model,
        use_cache=not args.no_cache,
        include_explanation=args.with_explanation,
        cache_dir=args.cache_dir,
        replay=args.replay,
    )

    input_path = args.input
    if args.held_out_only:
        # We swap in a smaller, filtered copy of the input file BEFORE
        # running anything, instead of filtering the results afterward.
        # That way, every part of the program downstream, the parsing,
        # the scoring, even the printed count of products, only ever
        # sees the held-out 10 products, and never accidentally sees the
        # other 40 that were used to write the rules.
        input_path = _write_held_out_subset(args.input)

    results, parse_errors, deadline_hit = run_file(
        input_path=input_path,
        output_path=args.output,
        config=config,
        concurrency=args.concurrency,
        max_wall_seconds=args.max_wall_seconds,
        limit=args.limit,
    )

    print_run_summary(results, replay=args.replay)
    if deadline_hit:
        print(
            f"\n[deadline] wall-time budget ({args.max_wall_seconds}s) was hit; "
            f"{args.output} is INCOMPLETE relative to the input, not a smaller success.",
            file=sys.stderr,
        )

    # We read and parse the input file again here, rather than reusing
    # something from run_file above. This function only needs the title
    # and the correct answers for each product, and reading it fresh
    # keeps this part of the code separate from run_file's own inner
    # workings.
    data = json.loads(Path(input_path).read_text())
    products, _ = load_products(data)
    comparisons = build_comparisons(results, products)
    if comparisons:
        report = compute_metrics(comparisons)
        print()
        print(format_report(report, header="Accuracy vs. ground truth"))
    else:
        # A fresh, unlabeled test file has NO correct answers included,
        # so this is the expected, quiet path for that file, not a sign
        # that something is broken.
        print("\n(no trusted_search_results field in this file, skipping accuracy metrics)")

    product_errors = [r for r in results if r.error]
    if args.replay:
        missing = [r for r in product_errors if r.error.startswith("ReplayMiss")]
        if missing:
            where = args.cache_dir if args.cache_dir is not None else "the default cache"
            print(f"REPLAY INCOMPLETE: {len(missing)} of {len(results)} products missing from {where}", file=sys.stderr)
    return 0 if not parse_errors and not product_errors and not deadline_hit else 1


def _write_held_out_subset(input_path: Path) -> Path:
    """See DECISIONS.md for the full reasoning: we sort all the product
    keys alphabetically, then take every 5th one (numbers 4, 9, 14, and
    so on). That gives us 10 out of 50 products held back and set aside.
    This is a fixed rule, not a random pick, so anyone reading this code
    can figure out exactly which 10 products get held out, with nothing
    hidden or random, and the same 10 come out every single time this
    runs on the same file."""
    data = json.loads(input_path.read_text())
    keys = sorted(data.keys())
    held_out_keys = keys[4::5]
    subset = {k: data[k] for k in held_out_keys}
    out_path = input_path.parent / f".held_out_subset_{input_path.name}"
    out_path.write_text(json.dumps(subset))
    return out_path


if __name__ == "__main__":
    raise SystemExit(main())
