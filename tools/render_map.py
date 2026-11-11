#!/usr/bin/env python3
"""Emit media/pipeline.svg, the path of one product through the pipeline.

The workers, retries and timeout printed on it are read from the code through tools/facts.py.
A cache key is a SHA-256 fingerprint of the model, prompts, schema, temperature and product text.
Replay is the keyless mode: answers come only from the cache and a miss is an error, never a call.

    python3 tools/render_map.py --write | --check
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facts  # noqa: E402
import palette as P  # noqa: E402

OUT = os.path.join(P.ROOT, "media", "pipeline.svg")
W, H = 1200, 700
M = 44
BW, BH, GAP = 250, 124, 36


def box(out, x, y, title, detail, col=P.EDGE, sw=2):
    out.append(P.rect(x, y, BW, BH, P.PANEL, col, sw=sw))
    out.append(P.t(x + 16, y + 42, title, 24, P.INK, bold=True, inner=BW - 32))
    out.append(P.t(x + 16, y + 82, detail, 23, P.MUTED, mono=True, inner=BW - 32))


def arrow(out, x1, y, x2, col=P.MUTED):
    out.append(P.line(x1, y, x2 - 2, y, col))
    out.append(f'<polygon points="{x2},{y} {x2 - 12},{y - 7} {x2 - 12},{y + 7}" fill="{col}"/>')


def build():
    c = facts.load()["code"]
    label = (f"Path of one product through the pipeline: input, parse, build prompt, cache key, API call, drop bad ids, "
             f"checkpoint, score. {c['workers']} workers, {c['retries']} retries, {c['timeout_s']} second timeout. "
             f"A dashed branch shows replay, where a miss stops and no call is made.")
    out = [P.svg_open(W, H, label)]
    out.append(P.rect(0, 0, 8, H, P.ACCENT))
    out.append(P.t(M, 56, "PIPELINE", 23, P.MUTED, spacing=2, inner=300))
    out.append(P.t(M, 112, "The path of one product", 44, P.INK, bold=True, inner=1100))
    row1 = [("Input JSON", "one record each"), ("Parse", "bad rows skipped"), ("Build prompt", "rules + results"), ("Cache key", "SHA-256")]
    row2 = [("API call", f"{c['retries']} retries, {c['timeout_s']} s"), ("Drop bad ids", "not in results"), ("Checkpoint", "after each one"), ("Score", "vs. labels")]
    y1, y2 = 160, 352
    xs = [M + i * (BW + GAP) for i in range(4)]
    for i, (t_, d) in enumerate(row1):
        box(out, xs[i], y1, t_, d, P.ACCENT if t_ == "Cache key" else P.EDGE, 3 if t_ == "Cache key" else 2)
        if i:
            arrow(out, xs[i - 1] + BW, y1 + BH // 2, xs[i])
    for i, (t_, d) in enumerate(row2):
        box(out, xs[i], y2, t_, d, P.ACCENT if t_ == "API call" else P.EDGE, 3 if t_ == "API call" else 2)
        if i:
            arrow(out, xs[i - 1] + BW, y2 + BH // 2, xs[i])
    # the miss path: from the cache key down and left to the API call
    cx = xs[3] + BW // 2
    ymid = y1 + BH + (y2 - y1 - BH) // 2
    out.append(P.line(cx, y1 + BH, cx, ymid))
    out.append(P.line(cx, ymid, xs[0] + BW // 2, ymid))
    out.append(P.line(xs[0] + BW // 2, ymid, xs[0] + BW // 2, y2 - 2))
    out.append(f'<polygon points="{xs[0] + BW // 2},{y2} {xs[0] + BW // 2 - 7},{y2 - 12} {xs[0] + BW // 2 + 7},{y2 - 12}" fill="{P.MUTED}"/>')
    out.append(P.t(cx - 12, ymid - 10, "miss", 23, P.MUTED, anchor="end"))
    # constants band
    by = y2 + BH + 28
    out.append(P.line(M, by, W - M, by))
    out.append(P.t(M, by + 34, f"{c['workers']} workers at a time. {c['retries']} retries. {c['timeout_s']} s timeout per call.", 24, P.INK, bold=True, inner=W - 2 * M))
    out.append(P.t(M, by + 68, "Cache hit: skip the call and go straight to Drop bad ids.", 23, P.MUTED, inner=W - 2 * M))
    # replay branch, dashed
    ry = by + 90
    out.append(P.rect(M, ry, W - 2 * M, 50, "none", P.ACCENT, dash="8 6"))
    out.append(P.t(M + 16, ry + 33, "Replay: a miss is a per-product error. No API call, no sleep.", 23, P.INK, inner=W - 2 * M - 32))
    out.append("</svg>\n")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(P.run(sys.argv[1:], OUT, build))
