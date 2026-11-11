"""Tests for src/metrics.py: precision, recall, contamination, coverage
and exact-set match, checked on small hand-made sets of predicted versus
labeled ids, including the empty and everything-rejected edge cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import ProductComparison, compute_metrics


def _cmp(key, predicted, truth):
    return ProductComparison(key=key, product_title=key, predicted=set(predicted), truth=set(truth))


def test_perfect_agreement_scores_1_everywhere():
    comparisons = [_cmp("a", [1, 2], [1, 2]), _cmp("b", [], [])]
    report = compute_metrics(comparisons)
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.exact_set_agreement_rate == 1.0
    assert report.product_contamination_rate == 0.0
    assert report.product_coverage_rate == 1.0


def test_false_positive_hurts_precision_and_flags_contamination():
    # truth is {1}, predicted {1, 2}: one true positive, one false positive
    comparisons = [_cmp("a", [1, 2], [1])]
    report = compute_metrics(comparisons)
    assert report.precision == 0.5
    assert report.recall == 1.0
    assert report.n_products_contaminated == 1
    assert report.product_contamination_rate == 1.0
    assert report.n_exact_match == 0


def test_false_negative_hurts_recall_not_precision():
    # truth is {1, 2}, predicted {1}: missed one, but everything predicted is correct
    comparisons = [_cmp("a", [1], [1, 2])]
    report = compute_metrics(comparisons)
    assert report.precision == 1.0
    assert report.recall == 0.5
    assert report.n_products_contaminated == 0


def test_coverage_denominator_excludes_products_with_no_ground_truth():
    # "a" has no truth at all (generic item, correctly rejected everything).
    # It must not count toward coverage's denominator or numerator.
    comparisons = [_cmp("a", [], []), _cmp("b", [1], [1, 2])]
    report = compute_metrics(comparisons)
    assert report.n_products_with_truth == 1  # only "b"
    assert report.n_products_covered == 1  # "b" got at least one right
    assert report.product_coverage_rate == 1.0


def test_all_reject_pipeline_does_not_look_good_on_coverage():
    # This is exactly the failure mode coverage exists to catch: precision
    # looks perfect (no false positives ever), but coverage is 0.
    comparisons = [_cmp("a", [], [1, 2]), _cmp("b", [], [3])]
    report = compute_metrics(comparisons)
    assert report.precision == 1.0  # no predictions at all, vacuously "precise"
    assert report.product_coverage_rate == 0.0


def test_disagreements_list_only_contains_non_exact_matches():
    comparisons = [_cmp("a", [1], [1]), _cmp("b", [1], [2])]
    report = compute_metrics(comparisons)
    assert len(report.disagreements) == 1
    assert report.disagreements[0].key == "b"


def test_empty_input_reports_are_vacuously_perfect_not_crashing():
    report = compute_metrics([])
    assert report.n_products == 0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.exact_set_agreement_rate == 1.0
