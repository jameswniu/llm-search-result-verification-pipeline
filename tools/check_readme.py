#!/usr/bin/env python3
"""Fails when README.md, its badges or its figures disagree with the live numbers.

Every number comes from tools/facts.py. Two groups exist. PRIVATE FILE numbers are parsed from
docs/REFEREE.md (the referee cards, one table per number). RECORDED SAMPLE numbers are recomputed
by a replay of answers recorded from one real model run. This script checks, one line each:

  1. figure files match their generators (the render scripts with --check)
  2. the README embeds the three figures and a GIF, and each figure carries its numbers as text
  3. required numbers appear in the README, and every decimal in prose is one facts.py prints
  4. the quick start output block equals what the replay run prints
  5. badge values match live values (the tests badge comes from a real pytest run)
  6. landing limits: words, paragraph and sentence length, badges, images, picture near the top

Prints one line per check and a final OK line, or exits 1.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import facts  # noqa: E402

README = (ROOT / "README.md").read_text()
FIGURES = {"media/hero.svg": "render_hero.py", "media/pipeline.svg": "render_map.py", "media/results.svg": "render_panel.py"}
BADGE_RE = re.compile(r"https://img\.shields\.io/badge/([A-Za-z_]+)-([^-\s)\"]+)-[A-Za-z0-9]+")
REQUIRED_BADGES = ("tests", "private_file_precision", "private_file_pooled_precision", "private_file_held_out_recall", "recorded_sample", "license")
fails: list[str] = []


def ok(msg: str) -> None:
    print(f"ok   {msg}")


def bad(msg: str) -> None:
    fails.append(msg)
    print(f"FAIL {msg}")


def prose_lines() -> list[str]:
    """README lines that count as prose: outside code fences, with HTML tags and code spans removed."""
    out, fenced = [], False
    for line in README.splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = re.sub(r"<[^>]+>", " ", line)
        line = re.sub(r"`[^`]*`", " ", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)
        out.append(line)
    return out


def check_figures() -> None:
    for fig, gen in FIGURES.items():
        r = subprocess.run([sys.executable, str(ROOT / "tools" / gen), "--check"], capture_output=True, text=True)
        (ok if r.returncode == 0 else bad)(f"figure {fig} matches {gen}" if r.returncode == 0 else f"figure {fig} drifted from {gen}\n{r.stdout[-800:]}")
        if f'src="{fig}"' not in README:
            bad(f"README does not embed {fig}")
        else:
            ok(f"README embeds {fig}")
    if 'src="media/demo.gif"' not in README or not (ROOT / "media" / "demo.gif").exists():
        bad("README must embed media/demo.gif and the file must exist")
    else:
        ok("README embeds media/demo.gif")


def svg_text(path: str) -> str:
    return " ".join(re.findall(r">([^<]+)</text>", (ROOT / path).read_text()))


def required(f: dict) -> dict[str, list[str]]:
    p, s, c = f["private"], f["sample"], f["code"]
    return {
        "README.md": [
            f"{p['held_precision_lo']} to {p['held_precision_hi']}", p["held_recall"], p["cv_precision"],
            f"[{p['cv_interval_lo']}, {p['cv_interval_hi']}]", f"{p['cv_recall_lo']} to {p['cv_recall_hi']}",
            p["keep_precision"], p["keep_recall"], f"{p['keep_trusted']} of {p['keep_total']}",
            f"{p['all_precision_lo']} to {p['all_precision_hi']}", p["all_recall"],
            f"About ${p['cost_rounded_per_100k']} per 100,000 products, estimated from recorded token counts at list prices",
            p["cost_calls"], p["cost_input_tokens"], p["cost_output_tokens"],
            f"precision {s['precision']} and recall {s['recall']} on {s['n_products']} products",
            f"cost ${s['cost_usd']}", f"missed {s['fn']} of {s['tp'] + s['fn']} right results",
        ],
        "media/hero.svg": [f"{p['held_precision_lo']} to {p['held_precision_hi']}", p["cv_precision"], f"{s['replayed']} of {s['n_products']}", "PRIVATE FILE", "RECORDED SAMPLE"],
        "media/results.svg": [f"{p['held_precision_lo']} to {p['held_precision_hi']}", p["held_recall"], f"{p['cv_precision']} [{p['cv_interval_lo']}, {p['cv_interval_hi']}]",
                              f"{p['cv_recall_lo']} to {p['cv_recall_hi']}", p["keep_precision"], p["keep_recall"], "PRIVATE FILE"],
        "media/pipeline.svg": [f"{c['workers']} workers", f"{c['retries']} retries", f"{c['timeout_s']} s timeout"],
    }


def check_numbers(f: dict) -> None:
    for path, needles in required(f).items():
        text = README if path == "README.md" else svg_text(path)
        missing = [n for n in needles if n not in text]
        (bad if missing else ok)(f"{path} is missing {missing}" if missing else f"{path} carries all {len(needles)} required numbers")
    allowed = {v for g in ("private", "sample") for v in map(str, f[g].values())}
    allowed |= {f"{float(v):.2f}" for v in f["private"].values() if re.fullmatch(r"\d+\.\d+", v)} | {"0.60", "0.15", "2.0", "3.12", "3.14"}  # 2.0 is Apache-2.0, 3.12 and 3.14 are Python versions
    stray = []
    for line in prose_lines():
        for tok in re.findall(r"\d+\.\d+", line):
            if tok not in allowed:
                stray.append(tok)
    (bad if stray else ok)(f"README prose has decimals facts.py does not print: {sorted(set(stray))}" if stray else "every decimal in README prose is printed by facts.py")
    for path in FIGURES:
        extra = {t for t in re.findall(r"\d+\.\d+", svg_text(path)) if t not in allowed and t not in ("0.5", "1.0")}
        (bad if extra else ok)(f"{path} shows decimals facts.py does not print: {sorted(extra)}" if extra else f"{path} decimals all come from facts.py")


def check_quickstart() -> None:
    m = re.search(r"ends with this summary, in under a second:\n\n```\n(.*?)\n```", README, re.S)
    if not m:
        return bad("README has no quick start output block")
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, "run.py", "sample/catalog.json", "--cache-dir", "sample/cache", "--replay", "--output", f"{tmp}/o.json"],
                           cwd=ROOT, capture_output=True, text=True, env=env)
    want, got = [x.rstrip() for x in m.group(1).splitlines()], [x.rstrip() for x in r.stdout.rstrip("\n").splitlines()]
    (ok if want == got and r.returncode == 0 else bad)("quick start output block equals the live replay output" if want == got and r.returncode == 0 else "quick start output block differs from the live replay output")


def pytest_count() -> str:
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    if r.returncode != 0 or re.search(r"\d+ failed|\d+ error", out):
        bad(f"pytest did not pass:\n{out[-600:]}")
        return "?"
    return str(sum(int(n) for n in re.findall(r"(\d+) passed", out)))


def check_badges(f: dict) -> None:
    found = {}
    for label, value in BADGE_RE.findall(README):
        found.setdefault(label, value.replace("_", " ").replace("%2F", "/"))
    for label in REQUIRED_BADGES:
        if label not in found:
            bad(f"required badge {label} is missing")
    p, s = f["private"], f["sample"]
    n = pytest_count()
    live = {
        "tests": f"{n} passed",
        "private_file_precision": f"{p['held_precision_lo']} to {p['held_precision_hi']}",
        "recorded_sample": f"{s['replayed']}/{s['n_products']} replayed",
        "private_file_pooled_precision": p["cv_precision"],
        "private_file_held_out_recall": p["held_recall"],
        "license": "Apache 2.0",
    }
    for label, want in live.items():
        if label in found:
            (ok if found[label] == want else bad)(f"badge {label} = {want}" if found[label] == want else f"badge {label} says {found[label]!r}, live is {want!r}")
    if len(re.findall(r"tests/. \| (\d+) tests", README)) != 1 or re.search(r"`tests/` \| (\d+) tests", README).group(1) != n:
        bad(f"Files table test count differs from the live count {n}")
    else:
        ok(f"Files table test count is {n}")


def check_limits() -> None:
    lines = prose_lines()
    words = sum(len(re.findall(r"[A-Za-z0-9$][\w$.,%'\[\]-]*", ln)) for ln in lines)
    paras, cur = [], []
    for ln in lines + [""]:
        if ln.strip().startswith("- "):
            if cur:
                paras.append(" ".join(cur))
            cur = [ln]
        elif ln.strip() and not ln.strip().startswith(("|", "---")):
            cur.append(ln)
        elif cur:
            paras.append(" ".join(cur)); cur = []
    longest_para = max((len(p.split()) for p in paras), default=0)
    sentences = [s for p in paras for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
    longest_sent = max((len(s.split()) for s in sentences), default=0)
    print(f"     words {words}, longest paragraph {longest_para} words, longest sentence {longest_sent} words")
    (ok if words <= 900 else bad)(f"at most 900 words ({words})")
    (ok if longest_para <= 70 else bad)(f"no paragraph over 70 words ({longest_para})")
    (ok if longest_sent <= 35 else bad)(f"no sentence over 35 words ({longest_sent})")
    badges = len(BADGE_RE.findall(README)) + README.count("badge.svg")
    (ok if badges >= 4 else bad)(f"at least 4 badges ({badges})")
    images = len(re.findall(r"<img ", README))
    (ok if images >= 3 else bad)(f"at least 3 images ({images})")
    first = [ln for ln in README.splitlines() if ln.strip()][:15]
    (ok if any("<img" in ln for ln in first) else bad)("a picture within the first 15 non-empty lines")
    gl = [ln for ln in README.splitlines() if ln.strip()][0]
    (ok if 'src="media/hero.svg"' in gl else bad)("the hero is line 1")


def main() -> int:
    f = facts.load()
    check_figures()
    check_numbers(f)
    check_quickstart()
    check_badges(f)
    check_limits()
    if fails:
        print(f"\n{len(fails)} check(s) failed")
        return 1
    print("\nOK: README, badges and figures match live output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
