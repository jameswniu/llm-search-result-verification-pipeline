#!/usr/bin/env python3
"""Emit media/results.svg, precision and recall with their ranges against the keep-everything baseline.

Every bar is a PRIVATE FILE number: measured on a private labeled file, recorded, and not
reproducible from this repo. Numbers are parsed from docs/REFEREE.md through tools/facts.py.
Precision is the share of the picked results that were right. Recall is the share of the right
results that were picked. A whisker is the range across repeated runs, or the 95% interval for the
pooled row. Keep-everything is the no-model baseline that picks every result.

    python3 tools/render_panel.py --write | --check
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facts  # noqa: E402
import palette as P  # noqa: E402

OUT = os.path.join(P.ROOT, "media", "results.svg")
W, H = 1200, 660
X0, PW = 400, 480
TINT = ' fill-opacity="0.25"'


def X(v):
    return round(X0 + PW * v, 1)


def bar(out, y, name, point, lo, hi, text, col):
    out.append(P.t(X0 - 16, y + 27, name, 23, P.MUTED, anchor="end"))
    out.append(P.rect(X0, y, round(PW * point, 1), 36, col, col, extra=TINT))
    cy = y + 18
    if hi > lo:
        out.append(P.line(X(lo), cy, X(hi), cy, P.INK, sw=3))
        out.append(P.line(X(lo), cy - 11, X(lo), cy + 11, P.INK, sw=3))
        out.append(P.line(X(hi), cy - 11, X(hi), cy + 11, P.INK, sw=3))
    out.append(P.t(X0 + PW + 24, y + 27, text, 24, P.INK, bold=True, inner=W - 40 - (X0 + PW + 24)))


def build():
    p = facts.load()["private"]
    fl = {k: float(v) for k, v in p.items() if k.startswith(("held_", "cv_", "keep_")) and k not in ("keep_trusted", "keep_total")}
    label = (f"Private file, recorded. Held-out 10: precision {p['held_precision_lo']} to {p['held_precision_hi']}, recall {p['held_recall']}. "
             f"Pooled over 4 folds: precision {p['cv_precision']} with interval {p['cv_interval_lo']} to {p['cv_interval_hi']}, "
             f"recall {p['cv_recall_lo']} to {p['cv_recall_hi']}. Keep everything: precision {p['keep_precision']}, recall {p['keep_recall']}.")
    out = [P.svg_open(W, H, label)]
    out.append(P.rect(0, 0, 8, H, P.ACCENT))
    out.append(P.t(40, 56, "PRIVATE FILE", 23, P.ACCENT, spacing=2, inner=300))
    out.append(P.t(40, 106, "Precision and recall, against keeping everything", 36, P.INK, bold=True, inner=1100))
    rows = [
        ("Held-out 10", "4 runs", (fl["held_precision_lo"] + fl["held_precision_hi"]) / 2, fl["held_precision_lo"], fl["held_precision_hi"],
         f"{p['held_precision_lo']} to {p['held_precision_hi']}", fl["held_recall"], fl["held_recall"], fl["held_recall"], p["held_recall"], P.ACCENT),
        ("Pooled, 4 folds", "50 products", fl["cv_precision"], fl["cv_interval_lo"], fl["cv_interval_hi"],
         f"{p['cv_precision']} [{p['cv_interval_lo']}, {p['cv_interval_hi']}]", (fl["cv_recall_lo"] + fl["cv_recall_hi"]) / 2, fl["cv_recall_lo"], fl["cv_recall_hi"],
         f"{p['cv_recall_lo']} to {p['cv_recall_hi']}", P.ACCENT),
        ("Keep everything", "no model", fl["keep_precision"], fl["keep_precision"], fl["keep_precision"], p["keep_precision"],
         fl["keep_recall"], fl["keep_recall"], fl["keep_recall"], p["keep_recall"], P.MUTED),
    ]
    y = 150
    for name, sub, pp, plo, phi, ptext, rp, rlo, rhi, rtext, col in rows:
        out.append(P.t(40, y + 30, name, 24, P.INK, bold=True, inner=250))
        out.append(P.t(40, y + 62, sub, 23, P.MUTED, inner=250))
        bar(out, y, "precision", pp, plo, phi, ptext, col)
        bar(out, y + 48, "recall", rp, rlo, rhi, rtext, col)
        y += 130
    out.append(P.line(X0, y - 24, X0 + PW, y - 24))
    for v in (0, .5, 1):
        out.append(P.t(X(v), y + 6, f"{v:.1f}" if v else "0", 23, P.MUTED, anchor="middle"))
    out.append(P.t(40, y + 50, "Whisker: range across runs, or the 95% interval for the pooled row.", 23, P.MUTED, inner=1100))
    out.append(P.t(40, y + 80, "Recorded on a private file. Not reproducible from this repo.", 23, P.MUTED, inner=1100))
    out.append("</svg>\n")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(P.run(sys.argv[1:], OUT, build))
