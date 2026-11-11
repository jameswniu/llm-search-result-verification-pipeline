#!/usr/bin/env python3
"""This file tests whether our whole APPROACH actually works, not just
whether our one hand-written set of rules happens to work on these 50
products. It does this by splitting the 50 labeled products into 4
groups, and for each group, learning a fresh set of rules using only 3
of the 4 groups, then testing those rules on the ONE group they've never
seen. This is one big file on purpose, splitting into groups, writing
rules for each group, scoring each group, checking the rules against
each other, and printing everything, so this one file reads top to
bottom, instead of jumping between a library file and a thin wrapper
around it.

THE CORE IDEA, IN PLAIN WORDS
  This pipeline has no trained model of its own to fine-tune, no numbers
  inside a neural network to adjust. The closest thing it has to
  something "trainable" is the plain-English rules list in
  src/prompts.py. So here, "training on a group of products" means:
  ask an AI model to read that group's products and their correct
  answers, and write a general set of rules from what it notices. Then
  we test those written-out rules on a DIFFERENT group of products that
  were never shown to the model while it was writing the rules.

TWO DIFFERENT TESTS, NEVER MIXED TOGETHER INTO ONE NUMBER
  1. run_cross_validation.        For each of the 4 groups, the rules
                                   are learned using ONLY that group's
                                   own 3/4 of the data. This gives an
                                   honest answer to "does this METHOD of
                                   learning rules actually work on data
                                   it's never seen".
  2. run_shipped_prompt_variance. The one fixed, hand-written set of
                                   rules that will actually be used gets
                                   tested on those same 4 groups. This
                                   tells us how much the score wobbles
                                   between different slices of data, but
                                   it is NOT a fair test of "never seen
                                   before", because whoever wrote those
                                   rules by hand already read almost all
                                   of this data first.
  See DECISIONS.md for the full argument on why these two numbers can
  never be reported as if they were the same kind of thing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI, APIConnectionError, APIError, RateLimitError

from src.cache import DiskCache
from src.metrics import ProductComparison, compute_metrics
from src.parsing import Product, load_products
from src.prompts import build_system_prompt
from src.selector import MAX_RETRIES, RETRY_BASE_DELAY_S, ProductResult, Selector, SelectorConfig, estimate_cost_usd

N_FOLDS = 4  # "fold" is the standard name for one of these groups; we split the data into 4 of them


class SynthesisFailed(RuntimeError):
    """We raise this when writing a new rules list, or checking the rules
    lists against each other, keeps failing even after several retries.

    When ASKING ABOUT ONE PRODUCT fails, we just skip that one product
    and keep going (see selector.py). But there's no smaller piece to
    fall back to here: if we can't successfully write a group's rules,
    that whole group can never be scored. So instead of quietly
    continuing with a broken or missing set of rules, we stop the entire
    run and say so loudly.
    """


def _call_json_schema_with_retries(client: OpenAI, model: str, system_prompt: str, user_prompt: str, schema: dict, schema_name: str) -> tuple[dict, "SimpleNamespace"]:
    """This does the exact same "try a few times if something briefly
    goes wrong" thing that Selector.select does in selector.py, pulled
    out here so both the rule-writing call and the rule-checking call
    can share it instead of each having their own copy. Before this
    existed, a brief network hiccup while writing the rules would crash
    the ENTIRE cross-validation run with a raw error message, exactly the
    kind of problem selector.py was already built to survive when
    scoring individual products."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,  # same reason as everywhere else in this project: we want the exact same answer every time we ask the exact same question
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}},
                timeout=60.0,  # longer than a normal per-product question (30s), because this question is much bigger (up to 38 whole products worth of text) and only happens once per group, not once per product
            )
            content = json.loads(response.choices[0].message.content)
            return content, response.usage
        except (RateLimitError, APIConnectionError, APIError) as exc:
            # Something briefly went wrong on OpenAI's side, worth trying again.
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_DELAY_S * (2**attempt))
                continue
        except (json.JSONDecodeError, KeyError, AttributeError) as exc:
            # The answer came back broken even though we asked for a
            # strict format. Trying again would probably just fail the
            # same way, so we stop retrying.
            last_exc = exc
            break
    raise SynthesisFailed(f"{schema_name} call failed after retries: {type(last_exc).__name__}: {last_exc}")


# This is the random-number seed used ONLY for the confidence-range
# estimate near the bottom of this file (the "bootstrap" section), not
# for splitting into groups. Splitting into groups needs no randomness at
# all (it's sorted names, taken in fixed order), so there's nothing to
# make repeatable there. This seed just makes sure the confidence range
# comes out exactly the same every time we run this file again.
BOOTSTRAP_SEED = 1234
BOOTSTRAP_ITERATIONS = 2000

# =====================================================================
# PART 1. Writing a fresh rules list from one group's own products and
# their correct answers.
# =====================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are writing the rules section of a prompt for a downstream LLM \
classifier. That classifier decides which web search results are the exact same retail product as a \
given product title and description, for a tax-categorization pipeline where a wrong match is worse \
than a missed one.

You will see a set of TRAINING products. For each one you get the product title, description, every \
search result's title, link, and snippet, and which result numbers a human trusted as the same product.

Write a numbered list of general rules that a classifier could apply to a DIFFERENT, unseen product \
and get the same kind of judgment right. Generalize the pattern, do not name any specific product, \
brand, or retailer from the training set; a rule that only works on these exact rows is worthless on \
next week's different 50 products. Cover things like: what kind of difference between a candidate \
result and the query product should be ignored versus should disqualify it, and when a product should \
get no trusted results at all. Base every rule on a pattern you can actually point to in the training \
examples, do not invent a rule the data does not support.

Return only the numbered rules text, nothing else."""

SYNTHESIS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"rules_block": {"type": "string", "description": "The numbered rules text, nothing else."}},
    "required": ["rules_block"],
    "additionalProperties": False,
}


def _synthesis_user_prompt(train_products: list[Product]) -> str:
    lines = ["Training products:\n"]
    for p in train_products:
        trusted = set(p.trusted_search_results or [])
        lines.append(f"### {p.product_title!r} (description: {p.product_description!r})")
        if not p.search_results:
            lines.append("  (no search results)")
            continue
        for r in p.search_results:
            tag = "TRUSTED" if int(r.result_id) in trusted else "not trusted"
            # We send the FULL title, link, and snippet, not just the
            # title. An earlier version of this only sent the title to
            # save a bit of money, but the instructions above already
            # tell the model it will see all three fields, and some
            # important clues (a recipe blog vs. a real store, a website
            # that signals the wrong country) only show up in the link
            # or the snippet, never the title.
            lines.append(f"  [{tag}] {r.title} | {r.link} | {r.snippet}")
        lines.append("")
    return "\n".join(lines)


class RuleSynthesizer:
    """Turns one group's own training products into a written-out rules
    list. One question per group. We save the answer, keyed on exactly
    which products were used, so running this same split again (say, a
    second time on the same data) costs nothing the second time."""

    def __init__(self, client: OpenAI | None = None, cache: DiskCache | None = None, model: str = "gpt-4o-mini", use_cache: bool = True):
        self.client = client or OpenAI()
        self.cache = cache if cache is not None else DiskCache()
        self.model = model
        self.use_cache = use_cache

    def synthesize(self, train_products: list[Product]) -> tuple[str, dict]:
        user_prompt = _synthesis_user_prompt(train_products)
        key_payload = json.dumps(
            {"model": self.model, "system": SYNTHESIS_SYSTEM_PROMPT, "user": user_prompt, "kind": "synthesis"},
            sort_keys=True,
        )
        key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()

        if self.use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                # We didn't ask the model anything new this time, so all
                # the usage numbers below are zero, and the caller's
                # running cost total stays accurate.
                return cached["rules_block"], {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "cache_hit": True}

        content, usage = _call_json_schema_with_retries(self.client, self.model, SYNTHESIS_SYSTEM_PROMPT, user_prompt, SYNTHESIS_RESPONSE_SCHEMA, "synthesis")
        rules_block = content["rules_block"]
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = estimate_cost_usd(self.model, input_tokens, output_tokens)
        if self.use_cache:
            self.cache.set(key, {"rules_block": rules_block})
        return rules_block, {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost, "cache_hit": False}


# =====================================================================
# PART 2. Splitting the 50 labeled products into 4 groups.
# =====================================================================


@dataclass
class Fold:
    fold_id: int
    train: list[Product]  # the 3/4 of the data this group's rules get WRITTEN from
    test: list[Product]  # the 1/4 of the data this group's rules get TESTED on, which they never saw while being written


def make_folds(products: list[Product], k: int = N_FOLDS) -> list[Fold]:
    """We split the 50 products into 4 test groups, but not by just
    slicing them evenly down the middle. There are 12 products in the
    data whose correct answer is "zero matches", and if those all
    happened to land in ONE group, that group's precision would look
    perfect for the wrong reason (nothing to get wrong), and another
    group would look artificially worse. So we sort the "has zero
    correct answers" products and the "has some correct answers"
    products SEPARATELY, and deal each of those two lists out evenly
    across the 4 groups, one at a time, like dealing cards. Because
    everything here is based on sorted names, not random chance, this
    exact same split comes out every single time, with no need to
    remember a random seed anywhere."""
    with_labels = [p for p in products if p.trusted_search_results is not None]
    zero = sorted((p for p in with_labels if len(p.trusted_search_results) == 0), key=lambda p: p.key)
    nonzero = sorted((p for p in with_labels if len(p.trusted_search_results) > 0), key=lambda p: p.key)

    # Deal each list out one product at a time, cycling through the 4
    # groups (0, 1, 2, 3, 0, 1, 2, 3, ...), so both lists end up spread
    # as evenly as possible across all 4 groups.
    test_buckets: list[list[Product]] = [[] for _ in range(k)]
    for stratum in (zero, nonzero):
        for i, p in enumerate(stratum):
            test_buckets[i % k].append(p)

    all_by_key = {p.key: p for p in with_labels}
    folds = []
    for fold_id in range(k):
        test = test_buckets[fold_id]
        test_keys = {p.key for p in test}
        # Everything that is NOT in this group's test list becomes its
        # training list. Because of how we built it, train and test can
        # never accidentally overlap, they're just two halves of the
        # exact same 50 products.
        train = [p for key, p in all_by_key.items() if key not in test_keys]
        folds.append(Fold(fold_id=fold_id, train=train, test=test))
    return folds


# =====================================================================
# PART 3. Actually running one group, and the two big tests this file
# does.
# =====================================================================


@dataclass
class FoldRunResult:
    fold_id: int
    rules_block: str
    comparisons: list[ProductComparison]
    results: list[ProductResult]
    synthesis_usage: dict


@dataclass
class CVResult:
    folds: list[FoldRunResult]
    pooled_comparisons: list[ProductComparison]
    elapsed_s: float
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int


def _score_fold_test_set(selector: Selector, fold: Fold) -> tuple[list[ProductResult], list[ProductComparison]]:
    """Both big tests below share this same step: score every product in
    one group's TEST list using whichever Selector (and therefore
    whichever rules) it's handed, then pair up each answer with that
    same product's real correct answer. This function doesn't know or
    care which of the two tests is calling it, only the caller decides
    which Selector (which rules) to use."""
    results = [selector.select(p) for p in fold.test]
    test_by_key = {p.key: p for p in fold.test}
    comparisons = [
        ProductComparison(
            key=r.key,
            product_title=r.product_title,
            predicted=set(r.llm_trusted_search_results),
            truth=set(test_by_key[r.key].trusted_search_results or []),
        )
        for r in results
    ]
    return results, comparisons


def run_cross_validation(products: list[Product], base_config: SelectorConfig, k: int = N_FOLDS, cache: DiskCache | None = None, client: OpenAI | None = None) -> CVResult:
    """TEST 1: does the METHOD actually work. For each group, we write
    that group's rules using ONLY that group's own training products,
    then score that group's test products, which those rules have never
    seen before. This gives us an honest answer to "if we hand this
    approach a brand new set of labeled examples, will it come up with
    rules that actually work on products it hasn't seen", which is
    different from asking "is our one hand-written rules list good"."""
    cache = cache if cache is not None else DiskCache()
    client = client or OpenAI()
    synthesizer = RuleSynthesizer(client=client, cache=cache, model=base_config.model, use_cache=base_config.use_cache)

    start = time.monotonic()
    fold_results: list[FoldRunResult] = []
    total_cost, total_in, total_out = 0.0, 0, 0

    for fold in make_folds(products, k=k):
        # STEP A. Write this group's OWN rules, using only its own training products.
        rules_block, synth_usage = synthesizer.synthesize(fold.train)
        total_cost += synth_usage["cost_usd"]
        total_in += synth_usage["input_tokens"]
        total_out += synth_usage["output_tokens"]

        # STEP B. Build a Selector whose instructions are THIS group's
        # freshly written rules, dropped into the exact same fixed
        # wrapping the normal, shipped rules use. That way, the ONLY
        # difference between this and the real pipeline is the rules
        # text itself, nothing else changes.
        fold_config = SelectorConfig(
            model=base_config.model,
            temperature=base_config.temperature,
            include_explanation=base_config.include_explanation,
            use_cache=base_config.use_cache,
            max_retries=base_config.max_retries,
            request_timeout_s=base_config.request_timeout_s,
            system_prompt=build_system_prompt(rules_block),
        )
        selector = Selector(config=fold_config, client=client, cache=cache)

        # STEP C. Score this group's TEST products, the ones its own
        # rules have never seen, using those exact rules.
        results, comparisons = _score_fold_test_set(selector, fold)
        for r in results:
            total_cost += r.cost_usd
            total_in += r.input_tokens
            total_out += r.output_tokens

        fold_results.append(FoldRunResult(fold_id=fold.fold_id, rules_block=rules_block, comparisons=comparisons, results=results, synthesis_usage=synth_usage))

    # Every one of the 50 products shows up in exactly ONE group's test
    # list (make_folds splits them up, never overlaps), so putting all 4
    # groups' answers together gives us one "never seen before" answer
    # per product, across all 50, with nobody scored twice and nobody
    # left out. tests/test_cross_validate.py checks this directly.
    pooled = [c for fr in fold_results for c in fr.comparisons]
    return CVResult(folds=fold_results, pooled_comparisons=pooled, elapsed_s=time.monotonic() - start, total_cost_usd=total_cost, total_input_tokens=total_in, total_output_tokens=total_out)


def run_shipped_prompt_variance(products: list[Product], base_config: SelectorConfig, k: int = N_FOLDS, cache: DiskCache | None = None, client: OpenAI | None = None) -> CVResult:
    """TEST 2: how much does the score wobble for the rules we're
    ACTUALLY shipping (the fixed, hand-written ones, never rewritten
    per group). This is NOT a "never seen before" test, whoever wrote
    those rules by hand had already read most of these 50 products. The
    one truly clean test of the shipped rules is the separate 10
    held-out products described in README.md; the only FULLY fresh test
    is the brand-new 50 products coming next week."""
    cache = cache if cache is not None else DiskCache()
    client = client or OpenAI()
    start = time.monotonic()

    # We build just ONE Selector here and reuse it for all 4 groups,
    # unlike run_cross_validation above. That's the whole point of this
    # test: nothing changes between groups, we're only checking how much
    # the SCORE changes just from testing on different slices of data.
    shipped_config = SelectorConfig(
        model=base_config.model,
        temperature=base_config.temperature,
        include_explanation=base_config.include_explanation,
        use_cache=base_config.use_cache,
        max_retries=base_config.max_retries,
        request_timeout_s=base_config.request_timeout_s,
        system_prompt=None,  # leaving this blank means "use the normal, shipped rules from prompts.py"
    )
    selector = Selector(config=shipped_config, client=client, cache=cache)

    fold_results: list[FoldRunResult] = []
    total_cost, total_in, total_out = 0.0, 0, 0
    for fold in make_folds(products, k=k):
        results, comparisons = _score_fold_test_set(selector, fold)
        for r in results:
            total_cost += r.cost_usd
            total_in += r.input_tokens
            total_out += r.output_tokens
        fold_results.append(
            FoldRunResult(fold_id=fold.fold_id, rules_block="(shipped prompt, not refit per fold)", comparisons=comparisons, results=results, synthesis_usage={"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "cache_hit": False})
        )

    pooled = [c for fr in fold_results for c in fr.comparisons]
    return CVResult(folds=fold_results, pooled_comparisons=pooled, elapsed_s=time.monotonic() - start, total_cost_usd=total_cost, total_input_tokens=total_in, total_output_tokens=total_out)


# =====================================================================
# PART 4. A confidence range around the combined (pooled) score.
# =====================================================================


def bootstrap_ci(comparisons: list[ProductComparison], metric: str, n_boot: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """This gives us a "how sure are we" range around one number, like
    precision, using a trick called BOOTSTRAPPING. Here's the plain-words
    version of how it works: we have our real list of products with
    their scores. We build a brand-new, made-up list of the SAME SIZE by
    picking randomly from the real list, allowing the same product to
    get picked more than once. We compute the score on this made-up
    list. We do this 2,000 times, and each time we get a slightly
    different score, just from random luck in which products got picked.
    We then look at the middle 95% of all those 2,000 scores. That
    middle range is our "95% confidence interval", it's basically saying
    "if we had a different, random-but-similar batch of 50 products, the
    real score would probably fall somewhere in this range". We use a
    FIXED random seed here so this range comes out exactly the same
    every time we run this file, instead of jittering around on every
    rerun, for the same reason we use temperature 0 when asking the
    model a question: a number that keeps changing on you is not a
    number you can stand behind."""
    if not comparisons:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(comparisons)
    values = []
    for _ in range(n_boot):
        sample = [comparisons[rng.randrange(n)] for _ in range(n)]  # pick n products at random, WITH repeats allowed, this is the standard bootstrap trick
        values.append(getattr(compute_metrics(sample), metric))
    values.sort()
    # Line up all 2,000 made-up scores from lowest to highest, then take
    # the value that's 2.5% of the way up and the value that's 97.5% of
    # the way up. Everything in between is the middle 95%.
    return (values[int(0.025 * n_boot)], values[int(0.975 * n_boot) - 1])


def _mean_and_range(values: list[float]) -> tuple[float, float, float]:
    return sum(values) / len(values), min(values), max(values)


# =====================================================================
# PART 5. Do the 4 separately-written rules lists actually agree with
# each other?
# =====================================================================

STABILITY_SYSTEM_PROMPT = """You will see four independently written rules blocks. Each was synthesized \
from a different 3/4 slice of the same 50-product training set for the same classification task, so the \
same underlying rule may appear in more than one block with different wording, and two blocks might even \
disagree about what the rule should say.

Identify the distinct RULE CONCEPTS across all four blocks (for example "how pack size or count should \
be treated" is one concept, "how a different flavor should be treated" is another). For each concept, \
list every fold (0-3) that has a rule about it, and for EACH of those folds say the STANCE that fold's \
rule takes using exactly one of these words: "ignore" (the difference does not disqualify a match), \
"require" (the candidate must match on this dimension or it is disqualified), or "conditional" (the \
fold's rule says it depends on something else). Match concepts by meaning, not exact wording, but do \
not merge two folds into the same concept if their stances actually disagree, that disagreement is \
exactly what this analysis needs to surface."""

STABILITY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "A short name for the rule concept."},
                    "mentions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "fold": {"type": "integer"},
                                "stance": {"type": "string", "enum": ["ignore", "require", "conditional"]},
                            },
                            "required": ["fold", "stance"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["concept", "mentions"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["concepts"],
    "additionalProperties": False,
}


def analyze_rule_stability(cv_result: CVResult, client: OpenAI | None = None, cache: DiskCache | None = None, model: str = "gpt-4o-mini", use_cache: bool = True) -> list[dict]:
    """One extra question to the model: look at all 4 of the separately
    written rules lists and figure out which ideas show up in ALL of
    them, agreeing with each other, and which ones only one group came
    up with (probably just noise from that one small slice of data).
    If a rule idea shows up in every group but they DISAGREE on it (one
    says "ignore size", another says "size must match"), we mark it as
    CONFLICTING, not as agreement, even though it technically showed up
    4 times. An earlier version of this code got that wrong and counted
    a straight-up contradiction as if it were 4-way agreement; the
    review that caught that mistake is written up in DECISIONS.md.

    ONE KNOWN LIMIT, also from that same review: the model itself
    decides what counts as "the same idea" across the 4 lists, and its
    instructions technically allow it to split one real idea into two
    separate entries when the groups disagree, instead of merging them
    and flagging the disagreement. That would make a real conflict LOOK
    smaller than it is, not bigger. We could fix this by building a
    stricter, code-based way of matching up ideas instead of trusting
    the model to do it, but we haven't, because this check is a nice
    extra piece of evidence, not the headline number, and DECISIONS.md
    says exactly that.
    """
    client = client or OpenAI()
    cache = cache if cache is not None else DiskCache()

    user_prompt = "\n\n".join(f"Fold {fr.fold_id} rules:\n{fr.rules_block}" for fr in cv_result.folds)
    key_payload = json.dumps({"model": model, "system": STABILITY_SYSTEM_PROMPT, "user": user_prompt, "kind": "stability"}, sort_keys=True)
    key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()

    if use_cache:
        cached = cache.get(key)
        if cached is not None:
            return cached["concepts"]

    content, _usage = _call_json_schema_with_retries(client, model, STABILITY_SYSTEM_PROMPT, user_prompt, STABILITY_RESPONSE_SCHEMA, "stability")
    if use_cache:
        cache.set(key, content)
    return content["concepts"]


def format_stability_table(concepts: list[dict], k: int = N_FOLDS) -> str:
    lines = ["| Rule concept | Folds and stance | Verdict |", "|---|---|---|"]
    for c in sorted(concepts, key=lambda c: -len(c["mentions"])):
        mentions = c["mentions"]
        stances = {m["stance"] for m in mentions}
        folds_and_stance = ", ".join(f"{m['fold']}={m['stance']}" for m in sorted(mentions, key=lambda m: m["fold"]))
        n = len(mentions)
        if len(stances) > 1:
            # We check this FIRST, before checking "did all 4 groups
            # mention it". A rule idea that all 4 groups mention but
            # actually DISAGREE about is not "agreement" just because
            # the count is 4, it's the opposite of agreement.
            verdict = "CONFLICTING, folds disagree on the stance, not corroboration"
        elif n == k:
            verdict = "stable, all folds agree"
        elif n >= k / 2:
            verdict = "majority agree"
        else:
            verdict = "single-fold, likely that fold's own sample"
        lines.append(f"| {c['concept']} | {folds_and_stance} | {verdict} |")
    return "\n".join(lines)


# =====================================================================
# PART 6. Turning all of the above into text a human can actually read.
# =====================================================================


def format_per_product_table(cv_result: CVResult) -> str:
    lines = ["| Fold | Product | Predicted | Truth | TP | FP | FN |", "|---|---|---|---|---|---|---|"]
    for fr in cv_result.folds:
        for c in sorted(fr.comparisons, key=lambda c: c.product_title):
            lines.append(f"| {fr.fold_id} | {c.product_title} | {sorted(c.predicted)} | {sorted(c.truth)} | {sorted(c.true_positives)} | {sorted(c.false_positives)} | {sorted(c.false_negatives)} |")
    return "\n".join(lines)


def format_cv_summary(cv_result: CVResult) -> str:
    lines = ["== Per-fold refit (auto-derived pipeline, out-of-sample) =="]
    fold_reports = []
    for fr in cv_result.folds:
        report = compute_metrics(fr.comparisons)
        fold_reports.append(report)
        lines.append(f"Fold {fr.fold_id} (n={report.n_products}): precision={report.precision:.3f} recall={report.recall:.3f} contamination={report.product_contamination_rate:.3f} coverage={report.product_coverage_rate:.3f} exact={report.exact_set_agreement_rate:.3f}")

    # The headline number is the AVERAGE across all 4 groups, never
    # whichever single group happened to score the best. Picking just
    # the best group would mean picking whichever group happened to get
    # an easy set of test products, and that lucky number would not
    # show up again on next week's brand-new test set.
    p_mean, p_min, p_max = _mean_and_range([r.precision for r in fold_reports])
    r_mean, r_min, r_max = _mean_and_range([r.recall for r in fold_reports])
    lines.append("")
    lines.append(f"Headline (mean across the 4 folds, never the best fold): precision={p_mean:.3f} (range {p_min:.3f}-{p_max:.3f}), recall={r_mean:.3f} (range {r_min:.3f}-{r_max:.3f}).")

    # The "pooled" number is a second, related way to look at the same
    # data: instead of averaging 4 separate scores, we treat all 50
    # products' one-and-only answers as ONE big combined list, and
    # compute precision and recall directly over that whole list. This
    # is also the exact list the confidence range above is built from.
    pooled = compute_metrics(cv_result.pooled_comparisons)
    p_ci = bootstrap_ci(cv_result.pooled_comparisons, "precision")
    r_ci = bootstrap_ci(cv_result.pooled_comparisons, "recall")
    lines.append("")
    lines.append(
        f"Pooled (n={pooled.n_products}, every product scored exactly once by a prompt that never saw it): "
        f"precision={pooled.precision:.3f} [{p_ci[0]:.3f}, {p_ci[1]:.3f}] recall={pooled.recall:.3f} [{r_ci[0]:.3f}, {r_ci[1]:.3f}] "
        f"contamination={pooled.product_contamination_rate:.3f} coverage={pooled.product_coverage_rate:.3f} exact={pooled.exact_set_agreement_rate:.3f}"
    )
    lines.append(f"95% bootstrap interval, {BOOTSTRAP_ITERATIONS} resamples over products, seed={BOOTSTRAP_SEED}.")
    lines.append("")
    lines.append(f"Cost: ${cv_result.total_cost_usd:.4f} total ({cv_result.total_input_tokens} input tokens, {cv_result.total_output_tokens} output tokens), {cv_result.elapsed_s:.1f}s wall time (4 synthesis calls + {sum(len(fr.results) for fr in cv_result.folds)} scoring calls).")
    return "\n".join(lines)


def format_shipped_variance_summary(cv_result: CVResult) -> str:
    lines = [
        "== Shipped-prompt variance under repeated holdout (fixed prompt, NOT refit per fold) ==",
        "This is variance under resampling, not a generalization estimate: the shipped prompt was",
        "written after reading the 40-product dev split, so a fold here that overlaps that split is",
        "scoring the prompt on data its author already saw. The clean check stays the original",
        "held-out 10 in README.md; the fully fresh check is next week's live test set.",
        "",
    ]
    precisions, recalls = [], []
    for fr in cv_result.folds:
        report = compute_metrics(fr.comparisons)
        precisions.append(report.precision)
        recalls.append(report.recall)
        lines.append(f"Fold {fr.fold_id} (n={report.n_products}): precision={report.precision:.3f} recall={report.recall:.3f} contamination={report.product_contamination_rate:.3f} coverage={report.product_coverage_rate:.3f} exact={report.exact_set_agreement_rate:.3f}")
    p_mean, p_min, p_max = _mean_and_range(precisions)
    r_mean, r_min, r_max = _mean_and_range(recalls)
    lines.append("")
    lines.append(f"Mean precision={p_mean:.3f} (range {p_min:.3f}-{p_max:.3f}), mean recall={r_mean:.3f} (range {r_min:.3f}-{r_max:.3f}).")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="4-fold cross-validation over the labeled training set.")
    parser.add_argument("input", type=Path, help="Path to the labeled products JSON (must carry trusted_search_results).")
    parser.add_argument("--output-dir", type=Path, default=Path("cv_output"), help="Where to write the rule blocks and the two report files.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-shipped-variance", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Export it first: `export OPENAI_API_KEY=...`", file=sys.stderr)
        return 2

    data = json.loads(args.input.read_text())
    products, parse_errors = load_products(data)
    for e in parse_errors:
        print(f"[skip] malformed record: {e}", file=sys.stderr)
    labeled = [p for p in products if p.trusted_search_results is not None]
    if len(labeled) < N_FOLDS:
        print(f"Only {len(labeled)} labeled products found, need at least {N_FOLDS} for a {N_FOLDS}-fold split.", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = DiskCache()
    config = SelectorConfig(model=args.model, use_cache=not args.no_cache)
    report_sections = []

    # ------------------------------------------------------------------
    # STEP 1. The main test: write fresh rules per group, score each
    # group on data those rules have never seen.
    # ------------------------------------------------------------------
    try:
        print(f"Running {N_FOLDS}-fold cross-validation on {len(labeled)} labeled products...\n")
        cv_result = run_cross_validation(labeled, config, cache=cache)
    except SynthesisFailed as exc:
        # If we can't write a set of rules for one group, that whole
        # group can't be scored. A cross-validation run with one broken
        # group is worse than no run at all, it LOOKS finished and it
        # isn't, so we stop here loudly instead of quietly continuing.
        print(f"\nCross-validation aborted: {exc}", file=sys.stderr)
        return 1

    cv_summary = format_cv_summary(cv_result)
    print(cv_summary)
    report_sections.append(cv_summary)

    # ------------------------------------------------------------------
    # STEP 2. Save the evidence: the 4 written-out rules lists, and the
    # full per-product comparison, so anyone can trace any number above
    # back to exactly which product and which rule caused it.
    # ------------------------------------------------------------------
    for fr in cv_result.folds:
        (args.output_dir / f"fold_{fr.fold_id}_rules.txt").write_text(fr.rules_block)
    table_path = args.output_dir / "per_product_comparison.md"
    table_path.write_text(format_per_product_table(cv_result))
    print(f"\nPer-fold rule blocks written to {args.output_dir}/fold_*_rules.txt")
    print(f"Per-product comparison table written to {table_path}")

    # ------------------------------------------------------------------
    # STEP 3. Check if the 4 groups' rules agree with each other. This
    # can fail on its own without ruining the numbers we already have
    # above, it's an extra piece of evidence, not the headline result.
    # ------------------------------------------------------------------
    if not args.skip_stability:
        try:
            concepts = analyze_rule_stability(cv_result, cache=cache, model=args.model, use_cache=not args.no_cache)
        except SynthesisFailed as exc:
            print(f"\nRule-stability analysis skipped, call failed: {exc}", file=sys.stderr)
            concepts = None
        if concepts is not None:
            stability_table = format_stability_table(concepts)
            print()
            print(stability_table)
            report_sections.append("== Rule-stability diagnostic ==\n\n" + stability_table)

    # ------------------------------------------------------------------
    # STEP 4. The second test: how much does the fixed, shipped set of
    # rules wobble across the same 4 groups.
    # ------------------------------------------------------------------
    if not args.skip_shipped_variance:
        shipped_result = run_shipped_prompt_variance(labeled, config, cache=cache)
        shipped_summary = format_shipped_variance_summary(shipped_result)
        print()
        print(shipped_summary)
        report_sections.append(shipped_summary)

    results_path = args.output_dir / "cv_results.md"
    results_path.write_text("\n\n".join(report_sections) + "\n")
    print(f"\nFull results written to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
