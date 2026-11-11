# Decisions

This is my decision log for this pipeline, built as a timed exercise. The labeled data it was developed on is private and not included. The pipeline that picks which web search results actually match a given product. One line each, what I did, why, and what I cut. Nothing here I can't defend out loud.

## Cuts

- I cut search, scraping and categorization. The scope was fixed up front: selection only. Building search or scraping is scope creep.
- No fine-tuning, embeddings or vector search. 50 rows train nothing, and one well-specified call is cheaper to build, explain and run at 100k.
- No product names in code. Every rule is prose in `src/prompts.py`, because a rule fit to these 50 is worthless on the next 50.
- I cut the front end. It was out of scope.
- `reason` and `confidence` sit behind `--with-explanation`, off. Extra fields outside `llm_trusted_search_results` are optional by the output contract, so off is my default, not a rule. The written contract is the bare key, and the extras are one flag away.

## Design

- One `gpt-4o-mini` call per product at temperature 0. Measured, 5 repeats at 0 gave identical sets, at 0.9 two of three borderline products flipped.
- Strict JSON schema output, not free text plus a parser. Removes the almost-valid-JSON failure class.
- Any id outside the product's result count is dropped in `src/selector.py`. Never happened here, one line, testable.
- Parse the inner `product_title` field, never the dict key. Keys carry trailing spaces and one concatenates title and description.
- A malformed record is skipped and logged, not fatal. A 100k-row input file will have bad rows.
- Disk cache keyed on model, prompts, schema and temperature. A prompt edit invalidates itself, and a rerun costs nothing.
- 8 threads, a wall-time budget, a checkpoint after every product. A run started in the session finishes inside it and keeps what finished.
- Retry transient vendor errors three times with backoff, never retry a parse failure. A parse failure under a schema is a bug, not a fault.
- Held out every 5th sorted key, 10 of 50. Reproducible from code alone, and never read while writing the prompt.
- Precision over recall in the prompt wording. A false positive taints a tax filing, a false negative costs one corroborating source.

## Fixed after code review

- Result text is labeled as scraped content and delimited per result, so an injected instruction reads as untrustworthy, not as a command.
- The cache key was missing temperature. Fixed and tested.
- `--max-wall-seconds` only stopped new work, then blocked on the queue. Now cancels unstarted work at the budget, proven with a fake slow selector.
- A crash in our code looked like an empty result. `run.py` now exits non-zero on any product error.
- Two cache writes could race on one temp file. Per-write unique names.
- A malformed record crashed the checkpoint writer. Passed through unchanged now.
- A truncated run reported success. The pipeline now returns a truncation flag and `run.py` exits non-zero on it.
- A hung call could outlive the budget because the executor joins threads at exit. Each call carries a 30 second request timeout.

## Cross-validation, two measurements, never merged

- Variance on the shipped prompt across 4 folds, mean precision 0.81, recall 0.57. Variance, not generalization, since I read 40 of the 50.
- Per-fold refit, rules synthesized from each fold's training labels only, held-out fold scored blind. Mean precision 0.80, recall 0.49. Pooled over 50, precision 0.79 [0.68, 0.89]. Evidence about the method, not the shipped prompt.
- The headline is the fold mean, never the best fold. The shipped prompt is the hand-written one in `src/prompts.py`, and the four fold prompts are the method being tested.
- Rule stability across the four synthesized blocks, 6 of 10 concepts unanimous with the same stance, 2 conflicting (ingredient list, packaging type), 3 from one to three folds. Unanimous ones are defended, the rest are the cut list.
- Fixes from review, synthesis now sees title, link and snippet like scoring does, stability reports a stance per fold so disagreement is CONFLICTING rather than agreement, synthesis and stability calls got retries and timeouts, `--no-cache` reaches all three caches.
- Open limit, the stability check's concept grouping is itself an LLM judgment and can split one concept into two. Not fixed, it is an exhibit, not the headline.
- I ran no configuration search. If I had, the winner's CV score would be biased upward, and the fresh 50 is the untouched check.
- Any prompt change made after reading a held-out product's error moves that product into development data. None made as of this build.

## Label noise, not a bug

- On one labeled product, a result reads as an exact match and the pipeline selects it, but the label does not. Surfaced, not hidden.

## Breaks performed, not predicted

| # | Break | Where | Measured |
|---|---|---|---|
| 1 | Read the dict key instead of `product_title` | Parsing, `parse_product` in `src/parsing.py` | 3 of 7 parsing tests fail, diff shows `'good key' == 'Good Product'` |
| 2 | Temperature 0 to 0.9 | Selector, `SelectorConfig` in `src/selector.py` | 3 borderline products, 5 runs each. Identical at 0, two of three flipped at 0.9 |
| 3 | Remove the size rule | Prompts, rule 1 in `src/prompts.py` | One product with a pack-size family drops from 5 selected ids to 3, losing the two other pack sizes |

The assumption row, what if size did change the category. Size becomes another variant word, which is how rule 8 is already written. One line, not a redesign, and break 3 is what it looks like.

## Known limits

- A failed call looks like an empty answer. When a model call fails, the output row carries `llm_trusted_search_results: []` next to `llm_error`. A reader that only looks at the result field takes a failure for "no matches". Check `llm_error` first, or drop the result field on errors. The accuracy summary also counts that empty answer, so a run with errors can look more accurate than it was.
- The time limit does not stop work in flight. After `--max-wall-seconds` passes, calls that already started keep retrying and can still spend API quota. A real deployment needs a deadline passed into the retries.
- Rule synthesis trusts the text it reads. `cross_validate.py` puts scraped titles, links and snippets into the prompt that writes the rules, without the warning the main prompt uses, so a hostile snippet could steer the rules. The recorded numbers came from this prompt as it is, so it is left unchanged and written down here.
- Cost is only known for one model. The price table holds gpt-4o-mini only, so a run with another `--model` reports a cost of $0 even though it is billed. Add the model's price before trusting the cost line.
- A product with no search results never reaches the model. It gets an empty answer in a normal run and in replay alike, so replay does not need a stored answer for it.
- A rerun starts over. The output file is rebuilt from the current run only, so rerunning after a timeout drops the earlier results for any product the new run does not finish. Cached answers make the redo cheap.
