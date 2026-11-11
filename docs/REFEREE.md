# Referee cards

One card per number a stranger will see. Regenerated from the code and the reports on 2026-09-27, three fresh no-cache runs against a private labeled file (`products.json`, not distributed), never typed from memory. If a live rerun during the review gives a slightly different number, that is expected and the reason is on every card below, not a bug to explain away.

## Mismatches

None outstanding. README.md quotes the same ranges as the cards below.

### 0.71 to 0.73 precision, 0.60 recall, the held-out 10 (the headline)

| | |
|---|---|
| What is claimed | 0.71 to 0.73 precision, 0.60 recall, on the 10 held-out products. |
| What is counted | Predicted `llm_trusted_search_results` ids against the human-labeled `trusted_search_results` list, for the 10 products never read while `src/prompts.py`'s rules were written. |
| How a match is decided | Exact integer id match, no partial credit and no overlap threshold, these are discrete catalog numbers, not bounding boxes. |
| Where the data came from | A private 50-product file, labeled by the data owner, not distributed with this repo. The held-out 10 are sorted product keys, every 5th one (indices 4, 9, 14, ...), a rule fixed in code before any prompt was written, see DECISIONS.md. |
| How to regenerate it | `python3 run.py products.json --held-out-only --output output_held.json --no-cache`; the match rule is `compute_metrics` in `src/metrics.py`. |
| Reproducible from this repo | No. It needs the private file. |
| The number that makes it look worse | With no model at all, keeping only each product's first three results already scores 0.59 precision on all 50 (the baselines are listed under the 4-fold card below). 0.71 to 0.73 on a slice the rules never saw is the real comparison, not a cherry-picked easy one. |
| Chosen before or after | The split (which 10 products) was fixed before any result was seen. The number is after, since that is what is being measured. |
| What this sample can and cannot say | 10 products is a small sample; a naive interval around it is wide. 4 repeated runs (0.706, 0.727, 0.727, 0.727) narrow the observed band tighter than that formal interval, but this is corroboration by repetition, not by size. The 4-fold pooled card below, at 50 products, is the sturdier estimate. |
| What moves it | One borderline call on one should-be-empty product, one borderline result, present in 1 of 4 measured runs and absent in the other 3. Every other prediction across all 4 runs is byte-identical. That single flip is the entire measured source of the range. |

Appears at README.md, in the top paragraph and the Numbers table.

### 0.78 to 0.80 precision, 0.56 recall, all 50 (includes development data)

| | |
|---|---|
| What is claimed | 0.78 to 0.80 precision, 0.56 recall, over all 50 products. |
| What is counted | Same exact-id match as above, over all 50 instead of the held-out 10. |
| How a match is decided | Same as above. |
| Where the data came from | Same file. 40 of the 50 products were read while writing the rules, so this measures fit to the training set, not generalization to a new one. Disclosed in README's split table and DECISIONS.md. |
| How to regenerate it | `python3 run.py products.json --output output.json --no-cache`. |
| Reproducible from this repo | No. It needs the private file. |
| The number that makes it look worse | This number is contaminated by development data; the held-out-10 card above is the one that should be trusted for anything resembling generalization, not this one. |
| Chosen before or after | After, and openly contaminated. That is why this row is labeled "includes development data" rather than presented as the headline. |
| What this sample can and cannot say | Fifty products is bigger than the held-out set, but the contamination means it cannot speak to generalization at all, only to whether the rules fit the set they were written against. |
| What moves it | 3 independent no-cache runs on 2026-09-27 measured precision at 0.780, 0.798, and 0.780 again; a roughly 2-point band from ordinary run-to-run model variance at temperature 0 on OpenAI's side, not from any code change (the code was untouched between runs). |

Appears at README.md, in the Numbers table.

### 0.79 precision [0.68, 0.89], the 4-fold refit, pooled (the out-of-sample estimate)

| | |
|---|---|
| What is claimed | 0.79 precision with a 95% bootstrap interval of [0.68, 0.89], recall 0.48 to 0.52, pooled over all 50. |
| What is counted | Same exact-id match, but each product's one prediction comes from a prompt whose rules were synthesized from the OTHER 3/4 of the data, so it never trained on its own label. |
| How a match is decided | Same as above. |
| Where the data came from | Same 50-product file, split into 4 folds stratified on whether a product has any trusted results at all, so the 12 zero-result products spread evenly instead of clumping in one fold (`make_folds` in `cross_validate.py`). |
| How to regenerate it | `python3 cross_validate.py products.json --output-dir cv_output --no-cache`. |
| Reproducible from this repo | No. It needs the private file. |
| The number that makes it look worse | Individual fold precision ranges 0.68 to 0.97 across the 4 folds (printed by `format_cv_summary`). The pooled 0.79 hides that spread; the worst single fold is the honest floor. |
| The comparison with no model | Same 50 products, same exact-id scoring, but the pick is fixed instead of made by the model. Keep every result: 0.53 precision, 1.00 recall (240 of the 455 results are labeled trusted, so 0.53 is what saying yes to everything scores). First result only: 0.60 precision, 0.13 recall. First three results: 0.59 precision, 0.37 recall. The 0.68 low end of the interval is still above all three. Each is a few lines over `products.json` with `compute_metrics`. |
| Chosen before or after | The fold split is fixed before any result is seen (sorted keys, stratified round robin, no stored seed). The bootstrap seed (1234) is fixed too, so the interval is reproducible given the same underlying predictions, though those predictions still carry the same run-to-run wobble as the other two cards. |
| What this sample can and cannot say | This is the only one of the three headline numbers that is a genuine out-of-sample estimate. It says the METHOD, rules learned automatically from labels, generalizes to unseen products. It does not say the shipped, hand-written prompt (written while reading 40 of the 50) will score this on a fresh set of 50 products; that is a separate, untested claim. |
| What moves it | 2 independent no-cache runs measured precision at 0.79 and 0.786, recall at 0.48 and 0.521, the same 2 to 4 point band as the other cards, same underlying cause. |

Appears at README.md, in the top paragraph and the Numbers table.

### About $24 per 100,000 products, the cost estimate

| | |
|---|---|
| What is claimed | About $24 per 100,000 products, estimated from recorded token counts at list prices. |
| What is counted | 352 recorded selection calls, 558,143 input tokens and 4,019 output tokens, which is a mean of 1,586 input and 11.4 output tokens per call. Only these totals are kept from the private run. |
| How a match is decided | Not a match, a sum. Input tokens at $0.15 per million and output tokens at $0.60 per million give $0.0861 in total, which is $24.47 per 100,000 products. |
| Where the data came from | Token counts stored with the answers from runs on the private 50-product file. The file and the answers are not distributed with this repo. |
| How to regenerate it | Sum the stored `input_tokens` and `output_tokens` over the private cache, then apply the two prices above. |
| Reproducible from this repo | No. It needs the private file. |
| The number that makes it look worse | List prices change, and a longer product description or more search results per product would raise the input tokens. The estimate holds for inputs the size of the private file. |
| Chosen before or after | After. The prices are the published list prices at the time of the runs. |
| What this sample can and cannot say | It prices the selection call only. It leaves out retries, search, scraping and any later step. |
| What moves it | Prompt length and result count. A cache hit costs nothing, so a rerun of the same file costs nothing. |

Appears at README.md, in the Results section.

## Why three cards show a range instead of a point

Temperature 0 sharply reduces but does not fully eliminate OpenAI's own run-to-run variance; this repo's own DECISIONS.md already measured this at a smaller scale (identical output across 5 repeats on 3 products, versus flips at temperature 0.9). What's new here is confirming it still shows up, in miniature, even at temperature 0, across separate full-dataset runs made minutes to days apart. The fix is not in this code, retrying or averaging would hide it, not explain it. The honest move is to report the observed band and name the one or two borderline products that actually move, which every card above does.
