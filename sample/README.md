# Sample

An invented catalog of 8 products, for trying the pipeline with no private data.

The products, brands and links are made up (the links use example.com). The labels are true by construction: each result was written to be a match or not, following the prompt's own rules. The answers are scripted by hand in `answers.json`, six exact and two imperfect on purpose. No model was called. The sample shows how a run reads its input, validates an answer, drops a made-up id and scores against labels. It is not model output and not a benchmark.

The labels follow the rules in `src/prompts.py`: size and pack count are ignored, flavor and formulation are not.

## Input shape

A JSON object of key to entry. The key is only a slot to write the answer back to, and trailing spaces in it are fine (one sample key has one). Read the title from the record, never the key.

- `product_title` (required): string.
- `product_description` (optional): string.
- `search_results` (required): an object of `"1"`, `"2"`, ... to a text block, `Title: ...\nLink: ...\nSnippet: ...`.
- `trusted_search_results` (optional): the correct ids as a list of integers. Absent means unlabeled. `[]` means labeled, and the correct answer is nothing.

## What the 8 products test

| Product | Correct ids | What it tests |
|---|---|---|
| Jasmine green tea, 20 ct | 1, 2, 3, 4 | A duplicate listing and a larger pack count. Mint and loose leaf do not count. |
| Cola zero sugar, 12 fl oz can | 1, 2, 6 | Size and pack count are ignored. Classic, diet and cherry do not count. |
| Pilsener lager, 6 pk | 1, 2 | Pack count ignored. A different style or non-alcoholic version does not count. |
| `Apple ` | None | A generic title has no specific product. Its key has a trailing space. |
| Vanilla protein powder, 2 lb | 1, 2, 7 | Size ignored, retailer duplicate counts. Ready to drink, bar and chocolate do not. |
| Abbreviated sourdough boule | 1, 2, 6 | The title is abbreviated. Size ignored, duplicate counts. |
| Gluten free pancake mix | 1, 2, 6 | A SKU string tells the regular mix apart from the gluten free one. |
| Blackberry sparkling water, 12 pk | None | Every result is a near miss. A confident-looking empty answer. |

## The two imperfect answers

- **Harbor Pilsener Lager 6 pk:** the scripted answer names id 9, which is not among the 6 results. The pipeline drops it and the run summary counts it.
- **FizzWell Cola Zero Sugar:** the scripted answer includes result 5, the cherry flavor, a near miss. It is a false positive, so the sample precision is below 1.00.

## Replay

```bash
python3 run.py sample/catalog.json --cache-dir sample/cache --replay --output output.json
python3 tools/check_replay.py
```

Replay serves answers only from `sample/cache`. It makes no call, needs no key and no `openai` package, and a product with no stored answer is an error that exits 1. The summary says the answers were scripted and prints no cost.

## Rebuild the cache

`sample/cache` is built from `answers.json` and `catalog.json`. A cache key is a fingerprint of the model name, prompts, schema, temperature and product text, so any change to those makes every entry miss. Rebuild with no key and no network.

```bash
python3 tools/build_sample_cache.py
```

Each file is marked `"source": "scripted"` with zero token counts. `tools/check_replay.py` fails if a file disagrees with `answers.json`, if a key no longer matches the prompt, or if a file is an orphan.
