"""P1 and word density, scored again by a committed script (post hoc), with the pre-roll audit.

Scorer written after the data. P1 (PRE-REGISTRATION 2026-09-23g) and word
density (PRE-REGISTRATION 2026-10-01d) were scored on 2026-10-01 with no
committed script (NOTES "2026-10-01 — PASS 1 COMPLETE. P1 IS FALSIFIED" and
"2026-10-01 — Word density scored"). Ali's decision, given in the session
on 2026-11-23 in his decisions on the step-2 pre-registration review (not
yet recorded in NOTES; his numbering, which differs from the 2026-11-20
decisions'): "6. P1 and word density (decision 2 of 11-22): (a) post-hoc
scorers that must reproduce the recorded numbers before their counts are
used."

The quantity: per VALID turn (not warm-up, not washout, with the four
marks), the recognizer's share of TTFA,
    (stt_final - vad_user_stopped) / (audio_out_first - speech_end_est),
its median per set (the median of the per-turn ratios; the ratio of the
medians printed beside it) and the median TTFA (not scored here: the
recorded TTFA CIs and the LLM-share column). Two percentile bootstrap CIs,
each 4000 resamples, seed 0 (stage_decomposition.boot_ci at 4b59f1e):
over turns, as recorded, and over items (every rep of a resampled item:
a turn index is the same file in each rep, so the turns are not
independent). The recorded points and n are reproduced; the turn-level CIs
reproduce 8 of the 9 recorded (pass 1n short_digit's lower bound is 46.5
against the recorded 46.4; the recorded script is not committed, so the
cause is not resolved). DIFF lines: the order differences, each set's items
resampled independently.

Sets and runs:
- P1, pass 1 (REACTIVE, three reps per set): sixteen (dev), short_digit,
  long_digit, multi_step;
- P1 re-scored on NOPARTIAL (pass 1n, 2026-10-02);
- word density: the 20-item single_step_reasoning set, three reps.

Every line names its items (the turn indices its turns fall on), each run's
build, and each turn dropped for a missing mark with the marks it lacks.

With --audit (preroll_audit.py --out): per set, the turns whose pre-roll
the audit flagged (stale, unverifiable) or could not check (unchecked),
named; the turns dropped for a missing mark, with their class; the median
share without the flagged ones (a post-hoc sensitivity); and the check the
item CIs rest on: each turn index is one played file (playback_timeline.json
at the audit's open chunk), the same in every rep, no file holding two.

    p1_score.py [--audit results/raw/run_logs/20261122_preroll_audit/turns.jsonl] [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

RAW = Path("results/raw/reactive")
SETS = {
    "P1 pass 1 (REACTIVE)": {
        "dev": ("20261001-135856-4c8175", "20261001-143936-138285", "20261001-152016-e39a27"),
        "short_digit": (
            "20261001-140103-5f8f0d",
            "20261001-144143-59f83f",
            "20261001-152239-3b6a33",
        ),
        "long_digit": (
            "20261001-140429-48340b",
            "20261001-144509-91261c",
            "20261001-152607-b814f4",
        ),
        "multi_step": (
            "20261001-140805-f3bafc",
            "20261001-144845-521874",
            "20261001-152945-09f625",
        ),
    },
    "P1 re-scored on NOPARTIAL (pass 1n)": {
        "dev": ("20261002-103129-d917b8", "20261002-111217-ca59d8", "20261002-115311-72786f"),
        "short_digit": (
            "20261002-103332-b34440",
            "20261002-111420-86da6c",
            "20261002-115514-ed6f32",
        ),
        "long_digit": (
            "20261002-103659-426fbe",
            "20261002-111804-8a8ad0",
            "20261002-115855-7562bf",
        ),
        "multi_step": (
            "20261002-104047-106290",
            "20261002-112138-a6c31a",
            "20261002-120244-4be4d4",
        ),
    },
    "word density": {
        "single_step_reasoning": (
            "20261001-163635-51ac10",
            "20261001-164213-1d8be7",
            "20261001-164736-f43dae",
        ),
    },
}
MARKS = ("vad_user_stopped", "stt_final", "audio_out_first", "speech_end_est")
BOOT = 4000  # stage_decomposition.boot_ci (4b59f1e), seed 0: reproduces 8 of the 9 recorded CIs


@dataclass
class Run:
    rows: list[tuple[str, int, float, float, float]]  # run, turn, share, ttfa_ms, stt_ms
    valid: int
    no_marks: list[tuple[int, list[str]]]  # valid turns dropped: (turn, the marks missing)
    nonpositive: int  # valid turns dropped for TTFA <= 0
    build: str
    config_hash: str


def read_run(run: str) -> Run:
    path = RAW / f"reactive-{run}" / "turns.jsonl"
    meta = json.loads(path.open().readline())
    out = Run([], 0, [], 0, str(meta.get("git_commit", "")), str(meta.get("config_hash", ""))[:8])
    for line in path.open():
        t = json.loads(line)
        if t.get("kind") != "turn_record" or not t.get("valid"):
            continue
        if t.get("warmup") or t.get("washout"):
            continue
        out.valid += 1
        st = t.get("stages_ms") or {}
        if not all(k in st for k in MARKS):
            out.no_marks.append((int(t["turn"]), [k for k in MARKS if k not in st]))
            continue
        ttfa = st["audio_out_first"] - st["speech_end_est"]
        if ttfa <= 0:
            out.nonpositive += 1
            continue
        stt = st["stt_final"] - st["vad_user_stopped"]
        out.rows.append((f"reactive-{run}", int(t["turn"]), stt / ttfa, ttfa, stt))
    return out


def median_ci(xs: list[float]) -> str:
    """Median and percentile bootstrap over turns (the recorded method)."""
    rng = random.Random(0)
    d = sorted(statistics.median([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(BOOT))
    lo, hi = d[int(0.025 * BOOT)], d[int(0.975 * BOOT) - 1]
    return f"{100 * statistics.median(xs):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def by_item(rows: list[tuple[str, int, float, float, float]]) -> list[list[float]]:
    """Shares grouped by item: a turn index is the same file in every rep."""
    g: dict[int, list[float]] = defaultdict(list)
    for _, turn, share, _, _ in rows:
        g[turn].append(share)
    return list(g.values())


def item_files(runs: tuple[str, ...], opens: dict[tuple[str, int], int]) -> str:
    """The played file of each located turn, checked to be one per turn index."""
    files: dict[int, set[str]] = defaultdict(set)
    for r in runs:
        tl = json.loads((RAW / f"reactive-{r}" / "playback_timeline.json").read_text())
        for (run, turn), o in opens.items():
            if run != f"reactive-{r}":
                continue
            hit = [str(path) for path, a, n in tl if int(a) <= o < int(a) + int(n)]
            if len(hit) != 1:
                raise SystemExit(f"{run} t{turn}: open chunk {o} in {len(hit)} files")
            files[turn].add(hit[0])
    split = sorted(t for t, f in files.items() if len(f) > 1)
    shared = len(files) - len({next(iter(f)) for f in files.values()})
    if split or shared:
        raise SystemExit(f"{runs}: turn indices {split} differ across reps; {shared} files shared")
    return f"{len(files)} located turn indices, each one file in every rep, no file holding two"


def item_draws(items: list[list[float]], rng: random.Random) -> list[float]:
    return [
        statistics.median([x for _ in items for x in items[rng.randrange(len(items))]])
        for _ in range(BOOT)
    ]


def item_ci(items: list[list[float]]) -> str:
    """Median, percentile bootstrap over items (all reps of a resampled item)."""
    d = sorted(item_draws(items, random.Random(0)))
    lo, hi = d[int(0.025 * BOOT)], d[int(0.975 * BOOT) - 1]
    pooled = [x for v in items for x in v]
    return f"{100 * statistics.median(pooled):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def diff_ci(a: list[list[float]], b: list[list[float]]) -> str:
    """Median share a - b, each set's items resampled independently."""
    rng = random.Random(0)
    da, db = item_draws(a, rng), item_draws(b, rng)
    d = sorted(x - y for x, y in zip(da, db, strict=True))
    lo, hi = d[int(0.025 * BOOT)], d[int(0.975 * BOOT) - 1]
    pa = statistics.median([x for v in a for x in v])
    pb = statistics.median([x for v in b for x in v])
    return f"{100 * (pa - pb):+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--audit", type=Path, default=None, help="preroll_audit.py --out")
    p.add_argument("--log", type=Path, default=None)
    args = p.parse_args()
    build = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if args.log and build.endswith("-dirty"):
        raise SystemExit(f"build {build} is dirty: a logged run needs a clean tree")
    cls: dict[tuple[str, int], str] = {}
    opens: dict[tuple[str, int], int] = {}
    if args.audit:
        for raw in args.audit.read_text().splitlines():
            r = json.loads(raw)
            cls[(str(r["run"]), int(r["turn"]))] = str(r["cls"])
            if r.get("open") is not None:
                opens[(str(r["run"]), int(r["turn"]))] = int(r["open"])
    lines = [
        f"# p1_score.py build {build}; {' '.join(sys.argv[1:])}; scorer written after the data"
    ]
    items: dict[tuple[str, str], list[list[float]]] = {}
    for block, sets in SETS.items():
        for name, runs in sets.items():
            got = [read_run(r) for r in runs]
            rows = [x for g in got for x in g.rows]
            shares = [x[2] for x in rows]
            items[(block, name)] = by_item(rows)
            ratio = statistics.median([x[4] for x in rows]) / statistics.median(
                [x[3] for x in rows]
            )
            dropped = [
                f"{r[-6:]} t{t} (no {', '.join(miss)})"
                + (f" {cls.get((f'reactive-{r}', t), 'not in the audit')}" if args.audit else "")
                for r, g in zip(runs, got, strict=True)
                for t, miss in g.no_marks
            ]
            line = (
                f"{block} {name}: n={len(rows)} turns on {len(items[(block, name)])} items "
                f"({sum(g.valid for g in got)} valid; dropped for a missing mark "
                f"{len(dropped)}: {', '.join(dropped) or 'none'}; for TTFA <= 0 "
                f"{sum(g.nonpositive for g in got)}); TTFA median "
                f"{statistics.median([x[3] for x in rows]):.0f} ms; stt share, median of "
                f"per-turn ratios, {median_ci(shares)} over turns, {item_ci(items[(block, name)])}"
                f" over items; ratio of medians {100 * ratio:.1f}%; build per run "
                f"{', '.join(f'{r[-6:]}@{g.build[:7]}' for r, g in zip(runs, got, strict=True))}"
                f", config {sorted({g.config_hash for g in got})}"
            )
            if args.audit:
                c = [cls.get((x[0], x[1]), "not in the audit") for x in rows]
                if c.count("not in the audit"):
                    raise SystemExit(
                        f"{name}: {c.count('not in the audit')} turns not in the audit"
                    )
                flagged = {"stale", "unverifiable"}
                named = [
                    f"{x[0][-6:]} t{x[1]} {k}" for x, k in zip(rows, c, strict=True) if k in flagged
                ]
                kept = [x for x, k in zip(rows, c, strict=True) if k not in flagged]
                located = {(x[0], x[1]): opens[(x[0], x[1])] for x in rows if (x[0], x[1]) in opens}
                unlocated = [f"{x[0][-6:]} t{x[1]}" for x in rows if (x[0], x[1]) not in opens]
                line += (
                    f"; pre-roll: stale {c.count('stale')}, unverifiable {c.count('unverifiable')},"
                    f" unchecked {c.count('unchecked')} ({', '.join(named) or 'none'}); without "
                    f"the flagged n={len(kept)} on {len(by_item(kept))} items, stt share "
                    f"{median_ci([x[2] for x in kept])} over turns, {item_ci(by_item(kept))} over "
                    f"items (post hoc); items: "
                    f"{item_files(runs, located)} (not located: {', '.join(unlocated) or 'none'})"
                )
            lines.append(line)
    p1, n1, wd = "P1 pass 1 (REACTIVE)", "P1 re-scored on NOPARTIAL (pass 1n)", "word density"
    for label, a, b in (
        ("P1 order, dev - short_digit (pass 1)", (p1, "dev"), (p1, "short_digit")),
        ("P1 order, dev - short_digit (pass 1n)", (n1, "dev"), (n1, "short_digit")),
        (
            "word density, single_step - dev (pass 1)",
            (wd, "single_step_reasoning"),
            (p1, "dev"),
        ),
    ):
        lines.append(
            f"DIFF {label}: {diff_ci(items[a], items[b])} over items "
            f"({len(items[a])} v {len(items[b])} items)"
        )
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
