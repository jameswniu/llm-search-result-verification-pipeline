"""This file does ONE thing well: take one product, ask the model about
it, and return a clean, priced, safe answer. Running this many times over
a whole file of products is a different file's job (pipeline.py).

THE WHOLE THING IN FOUR STEPS
  1. Build the question.  Combine the fixed instructions (or a custom
                          set of rules, during cross-validation) with
                          this one product's title, description, and
                          search results.
  2. Check if we already know the answer.  If we asked this EXACT same
                          question before, reuse that answer. No new
                          cost, no new wait, and the number can never
                          change on a rerun.
  3. Ask the model.       Always at temperature 0 (the "give me the same
                          answer every time" setting), in a strict format,
                          with a time limit, and a few retries if
                          something briefly goes wrong on OpenAI's side.
  4. Clean up and price it.  Estimate what the call cost, and throw away
                          any answer the model gave that doesn't actually
                          make sense, like pointing to a result number
                          that doesn't exist for this product.

REPLAY MODE
  With SelectorConfig.replay on, step 3 never happens. Answers come only
  from the cache, and a missing one becomes a per-product error that says
  the cache must be rebuilt. No key and no openai package is needed, which
  is why the import above is allowed to fail. _NoClient is a stand-in for
  the API client that raises if anything tries to use it. A stored answer
  marked "source": "scripted" was written by hand, not by a model.

WHY THE CROSS-VALIDATION CODE REUSES THIS SAME FILE
  `SelectorConfig.system_prompt` can be swapped out. The normal pipeline
  leaves it blank and gets the regular instructions from prompts.py.
  cross_validate.py sets it once per group to that group's own custom
  rules. Same retries, same caching, same cost tracking, either way,
  because it's the exact same code doing the asking.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    from openai import OpenAI, APIError, APIConnectionError, RateLimitError
except ImportError:  # replay mode needs no openai package
    OpenAI = None  # type: ignore[assignment,misc]

    class APIError(Exception):  # type: ignore[no-redef]
        pass

    class APIConnectionError(APIError):  # type: ignore[no-redef]
        pass

    class RateLimitError(APIError):  # type: ignore[no-redef]
        pass

from .cache import DEFAULT_CACHE_DIR, DiskCache, ReplayMiss, cache_key
from .parsing import Product
from .prompts import SYSTEM_PROMPT, build_user_prompt, response_schema

DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 1.5  # we wait 1.5s, then 3s, then 6s between tries, so a quick hiccup clears fast and a real outage doesn't get hammered

# The price OpenAI charges per 1 million tokens, as of when this was
# built. NOT LIVE, meaning it could be out of date; check
# platform.openai.com/pricing before quoting an exact dollar
# figure. This used to live in its own separate pricing.py file. We
# folded it into this file because it was short and only used in one
# place, and a repo with fewer separate files is easier to read.
PRICE_PER_1M_INPUT_TOKENS_USD = {"gpt-4o-mini": 0.15}
PRICE_PER_1M_OUTPUT_TOKENS_USD = {"gpt-4o-mini": 0.60}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price = PRICE_PER_1M_INPUT_TOKENS_USD.get(model)
    out_price = PRICE_PER_1M_OUTPUT_TOKENS_USD.get(model)
    if in_price is None or out_price is None:
        # If someone swaps in a different model above and forgets to add
        # its price here, we just report $0 instead of crashing. The
        # worst that happens is a wrong-looking cost line in the summary,
        # never a broken run.
        return 0.0
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


@dataclass
class ProductResult:
    key: str
    product_title: str
    llm_trusted_search_results: list[int]
    reason: str | None = None
    confidence: float | None = None
    error: str | None = None
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    scripted: bool = False  # the stored answer was written by hand, no model was called
    dropped_ids: list[int] = field(default_factory=list)  # ids the answer named that are not among the product's results


@dataclass
class SelectorConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.0  # 0 means "always give the same answer for the same question". We tested this: at 0.9, the same question got different answers each time; at 0, it never did
    include_explanation: bool = False  # off by default, since the output contract is llm_trusted_search_results only
    use_cache: bool = True
    max_retries: int = MAX_RETRIES
    # This limits how long we'll wait for a SINGLE question to come back.
    # A separate setting, --max-wall-seconds, limits the whole run, but
    # that setting only stops us from STARTING new questions once time is
    # up. Without a limit here too, one single slow question could hang
    # forever and blow past that whole-run time limit anyway.
    request_timeout_s: float = 30.0
    # Leaving this blank (None) means "use the normal, shipped
    # instructions". During cross-validation, we set this to a
    # custom-built set of rules for just that one group of products, so
    # the exact same class handles both the normal pipeline and
    # cross-validation.
    system_prompt: str | None = None
    # Where the disk cache lives. None means the default `.cache` folder.
    cache_dir: Path | None = None
    # Replay serves answers only from the cache. It never calls the API,
    # never sleeps and never writes. A missing answer becomes a
    # per-product error, so a run can never quietly drift from what was
    # stored.
    replay: bool = False


class _NoClient:
    """Stands in for the API client in replay mode. Any use raises."""

    def __getattr__(self, name: str):
        raise ReplayMiss(f"replay mode must not call the API (tried .{name})")


class Selector:
    def __init__(self, config: SelectorConfig | None = None, client: OpenAI | None = None, cache: DiskCache | None = None):
        self.config = config or SelectorConfig()
        if self.config.replay:
            self.client = client if client is not None else _NoClient()
            if cache is not None:
                self.cache = cache
            else:
                self.cache = DiskCache(self.config.cache_dir or DEFAULT_CACHE_DIR, read_only=True)
        else:
            if client is None:
                if OpenAI is None:
                    raise SystemExit("The openai package is not installed. Run: pip install -r requirements.txt")
                client = OpenAI()
            self.client = client
            if cache is not None:
                self.cache = cache
            else:
                self.cache = DiskCache(self.config.cache_dir) if self.config.cache_dir else DiskCache()

    def select(self, product: Product) -> ProductResult:
        cfg = self.config

        if not product.search_results:
            # There's nothing to pick from, so there's nothing to ask.
            # We don't even bother checking the cache; an empty list of
            # results has exactly one right answer: reject everything.
            return ProductResult(key=product.key, product_title=product.product_title, llm_trusted_search_results=[])

        # ------------------------------------------------------------------
        # STEP 1. Build the question and a short "fingerprint" that
        # uniquely identifies it.
        # ------------------------------------------------------------------
        system_prompt = cfg.system_prompt if cfg.system_prompt is not None else SYSTEM_PROMPT
        user_prompt = build_user_prompt(product.product_title, product.product_description, product.search_results)
        schema = response_schema(cfg.include_explanation)
        key = cache_key(cfg.model, system_prompt, user_prompt, schema, cfg.temperature)

        # ------------------------------------------------------------------
        # STEP 2. See if we already have this exact answer saved.
        # ------------------------------------------------------------------
        if cfg.use_cache or cfg.replay:
            cached = self.cache.get(key)
            if cached is not None:
                return self._to_result(product, cached, latency_s=0.0, cache_hit=True)
            if cfg.replay:
                return ProductResult(
                    key=product.key,
                    product_title=product.product_title,
                    llm_trusted_search_results=[],
                    error=(
                        f"ReplayMiss: no stored answer for {product.product_title!r} (key {key[:12]}). "
                        "The prompt, schema, model or product text changed since the cache was built. "
                        "Rebuild the sample cache with tools/build_sample_cache.py"
                    ),
                )

        # ------------------------------------------------------------------
        # STEP 3. Actually ask the model, with a few retries if something
        # briefly goes wrong.
        # ------------------------------------------------------------------
        start = time.monotonic()
        last_exc: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=cfg.model,
                    temperature=cfg.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "selection", "schema": schema, "strict": True},
                    },
                    # This is the per-question time limit explained above.
                    # Without it, one stuck question could hang around
                    # long after we thought the whole run was supposed to
                    # be finished.
                    timeout=cfg.request_timeout_s,
                )
                latency_s = time.monotonic() - start
                content = json.loads(response.choices[0].message.content)
                usage = response.usage
                payload = {
                    "trusted_result_ids": content.get("trusted_result_ids", []),
                    "reason": content.get("reason"),
                    "confidence": content.get("confidence"),
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                }
                if cfg.use_cache:
                    self.cache.set(key, payload)
                return self._to_result(product, payload, latency_s=latency_s, cache_hit=False)
            except (RateLimitError, APIConnectionError, APIError) as exc:
                # Something briefly went wrong on OpenAI's side (too many
                # requests, a dropped connection, etc). Our question was
                # fine, so it's worth trying again after a short wait.
                last_exc = exc
                if attempt < cfg.max_retries:
                    time.sleep(RETRY_BASE_DELAY_S * (2**attempt))
                    continue
            except (json.JSONDecodeError, KeyError, AttributeError) as exc:
                # The model's answer came back broken, even though we
                # asked for a strict format. This is not a "try again and
                # hope" kind of problem, asking the exact same question
                # again would likely just fail the exact same way, so we
                # give up on this ONE product and move on with the rest
                # of the run.
                last_exc = exc
                break

        latency_s = time.monotonic() - start
        return ProductResult(
            key=product.key,
            product_title=product.product_title,
            llm_trusted_search_results=[],
            error=f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown error",
            latency_s=latency_s,
        )

    def _to_result(self, product: Product, payload: dict, latency_s: float, cache_hit: bool) -> ProductResult:
        # ------------------------------------------------------------------
        # STEP 4. Figure out the cost, and remove any answer that doesn't
        # actually make sense.
        # ------------------------------------------------------------------
        cfg = self.config
        input_tokens = payload.get("input_tokens", 0)
        output_tokens = payload.get("output_tokens", 0)
        cost = 0.0 if (cache_hit and not cfg.replay) else estimate_cost_usd(cfg.model, input_tokens, output_tokens)  # reusing a saved answer didn't cost anything new, so it's free, however much it cost the FIRST time we asked
        valid_ids = {int(r.result_id) for r in product.search_results}
        # Forcing a strict answer FORMAT does not guarantee the answer
        # CONTENT makes sense. Nothing stops the model from returning
        # result number 12 on a product that only has 6 results. This
        # line is the only thing standing between a made-up number like
        # that and it ending up in our final output file.
        trusted = [i for i in payload.get("trusted_result_ids", []) if i in valid_ids]
        dropped = [i for i in payload.get("trusted_result_ids", []) if i not in valid_ids]
        return ProductResult(
            key=product.key,
            product_title=product.product_title,
            llm_trusted_search_results=sorted(trusted),
            reason=payload.get("reason") if cfg.include_explanation else None,
            confidence=payload.get("confidence") if cfg.include_explanation else None,
            latency_s=latency_s,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            cache_hit=cache_hit,
            scripted=payload.get("source") == "scripted",
            dropped_ids=dropped,
        )
