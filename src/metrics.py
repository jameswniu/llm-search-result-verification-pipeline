"""The five numbers this whole pipeline gets judged on. Nothing more.

THE FIVE NUMBERS, AND WHAT EACH ONE REALLY MEANS
  1. Precision.    Out of everything we said "yes" to, how much was
                    actually right. This is the main number, because a
                    wrong answer ends up in a real customer's tax filing.
  2. Recall.       Out of everything that SHOULD have gotten a "yes", how
                    much did we actually find. Less important here,
                    because finding just one or two good matches is
                    already enough to be useful.
  3. Contamination. Out of all the products, what share had at least one
                    wrong answer mixed in. This is the number a real
                    person at the company would actually notice and feel.
  4. Coverage.     Out of the products that DO have a correct answer,
                    what share did we find at least one of. This stops a
                    lazy pipeline (one that just says "no" to everything)
                    from looking great on precision while being useless.
  5. Exact match.  Out of all the products, what share did we get 100%
                    right, the whole set of correct answers and nothing
                    else. This is the hardest number to score well on,
                    and the easiest one for a person to just eyeball.

WHY THIS FILE NEVER SEES A PRODUCT NAME OR CALLS THE MODEL
  This file only ever looks at two lists of numbers per product: what we
  guessed, and what the real answer was. That is on purpose. The exact
  same code here scores one full run, ten held-out products, or one slice
  of a cross-validation split, because none of that changes what
  "precision" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProductComparison:
    key: str
    product_title: str
    predicted: set[int]
    truth: set[int]

    @property
    def true_positives(self) -> set[int]:
        return self.predicted & self.truth

    @property
    def false_positives(self) -> set[int]:
        return self.predicted - self.truth

    @property
    def false_negatives(self) -> set[int]:
        return self.truth - self.predicted

    @property
    def exact_match(self) -> bool:
        return self.predicted == self.truth

    @property
    def has_contamination(self) -> bool:
        return len(self.false_positives) > 0


@dataclass
class MetricsReport:
    n_products: int = 0
    total_predicted: int = 0
    total_truth: int = 0
    total_true_positives: int = 0
    total_false_positives: int = 0
    total_false_negatives: int = 0
    n_products_with_truth: int = 0  # how many products actually have a real answer; this is what "coverage" gets divided by
    n_products_covered: int = 0
    n_products_contaminated: int = 0
    n_exact_match: int = 0
    disagreements: list[ProductComparison] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.total_true_positives + self.total_false_positives
        # If we made zero guesses anywhere, we call precision a perfect
        # 1.0 instead of doing 0 divided by 0. That sounds too generous,
        # but it's actually correct: a pipeline that guessed nothing made
        # zero wrong claims, so its precision really is perfect. It's the
        # coverage number below that's supposed to punish it for finding
        # nothing, not this one.
        return self.total_true_positives / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.total_true_positives + self.total_false_negatives
        return self.total_true_positives / denom if denom else 1.0

    @property
    def product_contamination_rate(self) -> float:
        return self.n_products_contaminated / self.n_products if self.n_products else 0.0

    @property
    def product_coverage_rate(self) -> float:
        # We divide by n_products_WITH_TRUTH, never by all products. A
        # product that is SUPPOSED to have zero correct answers (a
        # generic item like "Apple") has nothing to "cover", and it
        # should not drag this number down just for correctly finding
        # nothing.
        return self.n_products_covered / self.n_products_with_truth if self.n_products_with_truth else 1.0

    @property
    def exact_set_agreement_rate(self) -> float:
        return self.n_exact_match / self.n_products if self.n_products else 1.0


def compute_metrics(comparisons: list[ProductComparison]) -> MetricsReport:
    report = MetricsReport()
    report.n_products = len(comparisons)
    for c in comparisons:
        report.total_predicted += len(c.predicted)
        report.total_truth += len(c.truth)
        report.total_true_positives += len(c.true_positives)
        report.total_false_positives += len(c.false_positives)
        report.total_false_negatives += len(c.false_negatives)
        if c.truth:
            # This product really does have a correct answer, so it
            # counts toward the coverage total below. Whether we actually
            # found any of that answer decides if it also counts as
            # "covered". A product with no correct answer at all never
            # gets here.
            report.n_products_with_truth += 1
            if c.true_positives:
                report.n_products_covered += 1
        if c.has_contamination:
            report.n_products_contaminated += 1
        if c.exact_match:
            report.n_exact_match += 1
        else:
            # We save every product that was NOT a perfect match, not
            # just count them, so we can print exactly which ones
            # disagreed and how, later.
            report.disagreements.append(c)
    return report


def format_report(report: MetricsReport, header: str = "Metrics") -> str:
    lines = [
        f"== {header} ({report.n_products} products) ==",
        f"Precision:            {report.precision:.3f}  (tp={report.total_true_positives}, fp={report.total_false_positives})",
        f"Recall:               {report.recall:.3f}  (tp={report.total_true_positives}, fn={report.total_false_negatives})",
        f"Product contamination: {report.product_contamination_rate:.3f}  ({report.n_products_contaminated}/{report.n_products} products picked up >=1 wrong site)",
        f"Product coverage:      {report.product_coverage_rate:.3f}  ({report.n_products_covered}/{report.n_products_with_truth} products with a trusted label got >=1 right site)",
        f"Exact-set agreement:   {report.exact_set_agreement_rate:.3f}  ({report.n_exact_match}/{report.n_products} products matched the label set exactly)",
    ]
    if report.disagreements:
        lines.append("")
        lines.append(f"Disagreements ({len(report.disagreements)} products):")
        for c in report.disagreements:
            lines.append(
                f"  {c.product_title!r}: predicted={sorted(c.predicted)} truth={sorted(c.truth)} "
                f"fp={sorted(c.false_positives)} fn={sorted(c.false_negatives)}"
            )
    return "\n".join(lines)
