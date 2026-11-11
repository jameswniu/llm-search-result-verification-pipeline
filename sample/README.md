# Sample

An invented catalog of 8 products, for trying the pipeline with no private data.

The products, brands and links are made up (the links use example.com). The labels are true by construction: each result was written to be a match or not, following the prompt's own rules. The answers in `cache/` were recorded from one real run of this pipeline against gpt-4o-mini on 2026-11-10, and they have not been edited since. The sample shows how a run reads its input, validates an answer and scores against labels. It is one run on eight invented products, not a benchmark.

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

## Where the recorded answers and the labels disagree

The model picked nothing wrong, so precision is 1.00. It missed 5 of the 18 right results, so recall is 0.72. Every miss is a different size or pack count of the right product, which the rules count as a match.

- **Jasmine green tea, 20 ct:** missed result 4, the 40 ct value pack.
- **Cola zero sugar, 12 fl oz can:** missed result 6, the 2 L bottle.
- **Pilsener lager, 6 pk:** missed result 2, the 12 pack.
- **Vanilla protein powder, 2 lb:** missed result 2, the 5 lb tub.
- **Sourdough boule, 24 oz:** missed result 2, the 16 oz loaf.

The answers stay exactly as recorded. The private-file numbers on the main page show the same lean, high precision with lower recall.

## Replay

```bash
python3 run.py sample/catalog.json --cache-dir sample/cache --replay --output output.json
python3 tools/check_replay.py
```

Replay serves answers only from `sample/cache`. It makes no call, needs no key and no `openai` package, and a product with no stored answer is an error that exits 1. The summary prices the stored calls from the token counts saved with each answer.

## Record it again

`sample/cache` holds one file per product, written by the run that recorded it. A cache key is a fingerprint of the model name, prompts, schema, temperature and product text, so any change to those makes every entry miss. Recording again needs a key, and the recorded run cost $0.0014 at list prices.

```bash
rm sample/cache/*.json
export OPENAI_API_KEY=...
python3 run.py sample/catalog.json --cache-dir sample/cache --output output.json
python3 tools/check_replay.py --pin
```

`recording.json` is the manifest of the recorded run: the model, the date it was pinned and a SHA-256 hash of each stored file. The last command above rewrites it. `tools/check_replay.py` fails if a product has no stored answer, if a key no longer matches the prompt, if a file is an orphan, or if a file differs from its pinned hash.

The recording is self-attested, which means you have this repo's word and no proof. The hashes show the files have not changed since they were pinned, and they catch an accidental edit. They cannot prove an answer came from a model.
