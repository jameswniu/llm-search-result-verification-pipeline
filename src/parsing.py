"""Turn the raw input JSON into clean, typed records. This file has one
job: the JSON arrives in a shape you cannot fully trust, and this file
fixes that before anything else touches it.

THE TRAP IN THE DATA
  The JSON is one big dictionary. Each entry has a key (a string) and a
  value (the actual product info). The KEY is not reliable. Some keys
  have extra spaces at the end. One key even smushes the title and the
  description together into one long string, even though the value's
  own `product_title` field is clean and correct. So the rule here is
  simple: always read the title, description, results, and labels from
  the VALUE, never from the key. We only keep the key around so we can
  write the answer back to the exact same slot in the output file.

WHY A BAD ROW NEVER STOPS THE WHOLE RUN
  A real input file with 100,000 products will have some broken rows in
  it somewhere, a missing field, a weird format, whatever. Both
  functions below are built to skip a bad row and keep going, instead of
  crashing the entire program because of one bad line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchResult:
    result_id: str  # the number "1" through "10", kept as text so it writes back out exactly as it came in
    raw: str  # the original "Title: ...\nLink: ...\nSnippet: ..." text, kept so we can see what went wrong if parsing fails
    title: str
    link: str
    snippet: str


@dataclass(frozen=True)
class Product:
    key: str  # the messy dictionary key from the input file; used ONLY so we can write the answer to the same spot
    product_title: str
    product_description: str
    search_results: list[SearchResult]
    # None means "this file has no correct answers in it at all" (this is
    # what the real test set will look like). An empty list means "this
    # product WAS labeled, and the correct answer really is zero results".
    # Those are two different things, and the scoring code treats them
    # differently.
    trusted_search_results: list[int] | None = field(default=None)


# This pattern looks for "Title: <something>", then a new line, then
# "Link: <something>", then a new line, then "Snippet: <everything else>".
# The snippet can contain almost anything (it's text copied from a real
# website), including more line breaks, so we let it run all the way to
# the end instead of stopping at the first line break inside it.
_LINE_RE = re.compile(
    r"Title:\s*(?P<title>.*?)\s*\nLink:\s*(?P<link>.*?)\s*\nSnippet:\s*(?P<snippet>.*)",
    re.DOTALL,
)


def parse_search_result(result_id: str, raw: str) -> SearchResult:
    """Split one "Title: ...\\nLink: ...\\nSnippet: ..." block of text into
    its three parts.

    If the text does not look like that (something is missing or broken),
    we do NOT crash. We just return a SearchResult with empty title and
    link, and put the whole raw text into the snippet field. One badly
    formatted result should never ruin the other nine results for the
    same product.
    """
    match = _LINE_RE.match(raw.strip())
    if not match:
        return SearchResult(result_id=result_id, raw=raw, title="", link="", snippet=raw.strip())
    return SearchResult(
        result_id=result_id,
        raw=raw,
        title=match.group("title").strip(),
        link=match.group("link").strip(),
        snippet=match.group("snippet").strip(),
    )


def parse_product(key: str, value: dict) -> Product:
    """Build one Product from one entry in the big input dictionary.

    Raises an error if the entry is truly unusable, no title to match
    against, or no search results to pick from. Whoever calls this
    (`load_products`, right below) catches that error for each entry on
    its own, so one bad entry becomes one skipped entry, never a crash
    that takes down the whole file.
    """
    if "product_title" not in value:
        raise ValueError(f"record for key {key!r} is missing product_title")
    if "search_results" not in value or not isinstance(value["search_results"], dict):
        raise ValueError(f"record for key {key!r} is missing search_results")

    # We sort using the NUMBER value of each id, not plain text sorting.
    # Plain text sorting would put "10" right after "1" (before "2"),
    # which is wrong. We want normal counting order: 1, 2, 3, ..., 10.
    results = [
        parse_search_result(rid, raw)
        for rid, raw in sorted(value["search_results"].items(), key=lambda kv: int(kv[0]))
    ]
    trusted = value.get("trusted_search_results")
    return Product(
        key=key,
        # We already checked that "product_title" is present above, but
        # .get(..., "") also protects against the field existing with a
        # blank (null) value in the JSON.
        product_title=value.get("product_title", ""),
        product_description=value.get("product_description", ""),
        search_results=results,
        # We convert every item to a real integer, one at a time. Some
        # source files store these numbers as text ("1" instead of 1),
        # and later code compares them directly against numbers, so this
        # is the one place where we make sure that comparison will work.
        trusted_search_results=[int(x) for x in trusted] if trusted is not None else None,
    )


def load_products(data: dict) -> tuple[list[Product], list[str]]:
    """Go through every entry in the input dictionary and parse it.

    Returns two lists: the products that parsed fine, and a list of error
    messages for the ones that did not. A broken entry never crashes the
    program, it just gets added to the error list and skipped. This is
    the one place where "the file might be messy" turns into "everything
    after this point can trust the data completely".
    """
    products: list[Product] = []
    errors: list[str] = []
    for key, value in data.items():
        try:
            products.append(parse_product(key, value))
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append(f"{key!r}: {exc}")
    return products, errors
