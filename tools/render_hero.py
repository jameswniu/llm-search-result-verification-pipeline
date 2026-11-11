#!/usr/bin/env python3
"""Emit media/hero.svg, the three headline cards at the top of the README.

Two cards are PRIVATE FILE numbers (measured on a private labeled file, recorded, parsed from
docs/REFEREE.md). One card is a RECORDED SAMPLE number (a replay of answers recorded from one real model run).
Every number comes from tools/facts.py.

    python3 tools/render_hero.py --write    regenerate
    python3 tools/render_hero.py --check    exit 1 with a diff if the committed file drifted
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facts  # noqa: E402
import palette as P  # noqa: E402

OUT = os.path.join(P.ROOT, "media", "hero.svg")
W, H = 1200, 470
M = 48


def build():
    f = facts.load()
    p, s = f["private"], f["sample"]
    cards = [
        (f"{p['held_precision_lo']} to {p['held_precision_hi']}", ["precision, 10 held-out", "products, 4 runs"], "PRIVATE FILE", P.ACCENT),
        (p["cv_precision"], ["precision, pooled over", f"{f['code']['folds']} folds of 50 products"], "PRIVATE FILE", P.ACCENT),
        (f"{s['replayed']} of {s['n_products']}", ["sample products replay", "with no key"], "RECORDED SAMPLE", P.GOOD),
    ]
    label = (f"Search result verification pipeline. Private file, recorded: precision {p['held_precision_lo']} to {p['held_precision_hi']} "
             f"on 10 held-out products and {p['cv_precision']} pooled over {f['code']['folds']} folds. "
             f"Recorded sample: {s['replayed']} of {s['n_products']} products replay with no key.")
    out = [P.svg_open(W, H, label)]
    inner = W - 2 * M
    out.append(P.rect(0, 0, 8, H, P.ACCENT))
    out.append(P.t(M, 54, "SEARCH RESULT VERIFICATION PIPELINE", 23, P.MUTED, spacing=2, inner=inner))
    out.append(P.t(M, 112, "Keep only the search results", 44, P.INK, bold=True, inner=inner))
    out.append(P.t(M, 164, "that are for the exact product", 44, P.INK, bold=True, inner=inner))
    out.append(P.t(M, 212, "One model call per product. An empty answer is a valid answer.", 25, P.MUTED, inner=inner))
    top, ch, gap = 244, 168, 24
    w = (inner - 2 * gap) // 3
    x = M
    for big, cap, tag, col in cards:
        out.append(P.rect(x, top, w, ch, P.PANEL, P.EDGE))
        out.append(P.rect(x, top, w, 4, col))
        out.append(P.t(x + 22, top + 40, tag, 23, col, spacing=1, inner=w - 44))
        out.append(P.t(x + 22, top + 98, big, 40, P.INK, bold=True, inner=w - 44))
        for i, line in enumerate(cap):
            out.append(P.t(x + 22, top + 130 + i * 28, line, 23, P.MUTED, inner=w - 44))
        x += w + gap
    out.append(P.t(M, 452, "python3 run.py sample/catalog.json --cache-dir sample/cache --replay", 23, P.MUTED, mono=True, inner=inner))
    out.append("</svg>\n")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(P.run(sys.argv[1:], OUT, build))
