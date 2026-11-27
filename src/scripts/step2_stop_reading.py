"""Step 2: the reading of the loops' P and stop records (registered in 0a3d1b0).

The draft reading registered in NOTES (2026-11-27, "RUN (2) ... the draft
reading of the loops' P and stop records", 0a3d1b0), approved by Ali with
the GO for run (2) ("2 (a) — GO for run (2) as is"; fed0e21), committed
with this script before it is first run. Descriptive: no verdict, and it
changes none.

Per offered hypothesis that the runaway guard keeps and that has a scored
fresh word (twl.step2_stops, which defines and checks each class): ITS STOP
(START, INSIDE, END), ITS CONTEXT (FROM THE OPEN; AFTER A COMMIT, as
MID-PHRASE, AT THE BUFFER'S END or AT A SEGMENT END) and ITS FIRST WORD (P,
P < tau, survival).

Lines per run, each share with a 95 % percentile bootstrap CI over files
(2000 resamples, seed 0, one set of draws per run); a resample that leaves a
group empty has no share: a NOTE line counts such resamples, and the CI is
over the rest:
- R1: the share of START stops FROM THE OPEN and AFTER A COMMIT, and their
  difference (after - open).
- R2 (run (2) only): among hypotheses AFTER A COMMIT, the counts and the
  share of START stops in each of the three classes; and the test's
  difference: MID-PHRASE and AT THE BUFFER'S END together, minus AT A
  SEGMENT END.
- R3: in each group of R1 and R2 (and the pooled MID-PHRASE and BUFFER'S END
  group), its count, the first word's median P (descriptive, no CI), the
  share with P < tau and the share that survives (each with its CI); and,
  beside it, how many first words have no number token (they do not
  survive here, unlike in the edge diagnostic: twl.step2_stops).

THE READING OF ALI'S HYPOTHESIS (item 9: "after a model commit that ends
mid-phrase, the next buffer begins with a cut word of low P, and the prefix
rule blocks everything behind it"), on run (2):
- SUPPORTED (descriptively) when R2's difference in START stops (MID-PHRASE
  and BUFFER'S END together, minus SEGMENT END) has a CI entirely above 0
  AND the same difference in the first word's share with P < tau has a CI
  entirely above 0.
- NOT SUPPORTED when either CI does not lie entirely above 0, and NOT
  SUPPORTED (not computable) when a group is empty or a CI cannot be
  computed (taken strictly, at coding time: any resample that leaves either
  group of R2's difference empty); a NaN never decides the reading. NOT SUPPORTED says the
  hypothesis is not shown on this stream, not that it is false.
- Beside it: R1 on the segment loop, where no commit ends mid-phrase.
Neither result is a verdict, and neither changes one; the association is not
the cause.

Under twl.step2_stops' checks START holds exactly when the first scored
word's P < tau, so R2's two differences are one quantity: the log prints
both, as registered.

VALIDITY:
- run (2): its header (a clean build; --reproduce of the retrained loop of
  a6ac6d5; no segment rule; not a smoke run); its driver log's DONE line for
  574 files and no STOP line; its commits, turn by turn, are the earlier
  run's (what the reproduction guard checked); 574 files and 15220 scored
  final words.
- the segment loop: its header (a clean build; the segment rule; not a smoke
  run); 574 files and 15220 scored final words.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_stop_reading.py
        --run2 results/raw/step2/closed_loop_rerun/closed_loop.jsonl
               results/raw/run_logs/20261127_closed_loop_rerun_full/driver.log
        --segments results/raw/step2/closed_loop_segments/closed_loop.jsonl
        [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.la_operating_point import BOOT
from scripts.step2_closed_loop_score import FILES, WORDS, recorded
from twl.step2_count import prepare_turn
from twl.step2_stops import Offer, offers_of_turn

AFTER = ("MID-PHRASE", "BUFFER'S END", "SEGMENT END")


@dataclass(frozen=True)
class Share:
    """A group's share: its value per resample, its value, its offers."""

    boot: list[float]
    point: float
    n: int


POOLED = "MID-PHRASE OR BUFFER'S END"


def group_of(name: str) -> Callable[[Offer], bool]:
    """The reading's groups, by name."""
    if name == "FROM THE OPEN":
        return lambda o: o.context == "OPEN"
    if name == "AFTER A COMMIT":
        return lambda o: o.context != "OPEN"
    if name == POOLED:
        return lambda o: o.context in ("MID-PHRASE", "BUFFER'S END")
    if name in AFTER:
        return lambda o: o.context == name
    raise ValueError(name)


def pct(vals: Sequence[float]) -> tuple[float, float]:
    v = sorted(x for x in vals if x == x)
    if not v:
        return float("nan"), float("nan")
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def ratio(num: Sequence[int], den: Sequence[int], ix: Sequence[int]) -> float:
    d = sum(den[i] for i in ix)
    return sum(num[i] for i in ix) / d if d else float("nan")


def load(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    recs = [json.loads(x) for x in run.read_text().splitlines() if x.strip()]
    return recs[0]["header"], recs[1:]


def arg_of(header: dict[str, Any], flag: str) -> str:
    a = [str(x) for x in header["args"]]
    return a[a.index(flag) + 1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run2", nargs=2, type=Path, required=True, metavar=("RUN", "DRIVER_LOG"))
    p.add_argument("--segments", type=Path, required=True, help="the segment loop's output")
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
    lines = [f"# step2_stop_reading.py build {build}; {' '.join(sys.argv[1:])}"]

    def say(line: str) -> None:
        lines.append(line)
        print(line, flush=True)

    run2, driver_log = args.run2
    h2, files2 = load(run2)
    hs, files_s = load(args.segments)
    for name, h in (("run (2)", h2), ("the segment loop", hs)):
        if str(h["build"]).endswith("-dirty"):
            raise SystemExit(f"{name}: the run's build {h['build']} is dirty")
        if h.get("limit"):
            raise SystemExit(f"{name}: a smoke run (limit {h['limit']})")
    if not h2.get("reproduce") or h2.get("segments"):
        raise SystemExit("run (2): not a --reproduce run of the plain rule")
    if not hs.get("segments"):
        raise SystemExit("the segment loop: its header has no segment rule")
    drv = Path(driver_log).read_text()
    if re.search(r"^STOP", drv, re.M) or not re.search(rf"^DONE: {FILES} files in ", drv, re.M):
        raise SystemExit(f"run (2): {driver_log} has a STOP line or no DONE line for {FILES} files")
    h_old, files_old = load(Path(str(h2["reproduce"])))
    if len(files_old) != len(files2):
        raise SystemExit("run (2): its files are not the earlier run's")
    for r, o in zip(files2, files_old, strict=True):
        if len(r["turns"]) != len(o["turns"]) or any(
            recorded(t) != recorded(u) for t, u in zip(r["turns"], o["turns"], strict=True)
        ):
            raise SystemExit("run (2): its commits differ from the earlier run's")
    say(
        f"VALIDITY run (2): {run2} (build {str(h2['build'])[:7]}, clean; not a smoke run); "
        f"{driver_log}: DONE for {FILES} files, no STOP line; its commits, turn by turn, are "
        f"{h2['reproduce']}'s (build {str(h_old['build'])[:7]})"
    )

    def read_run(
        label: str, header: dict[str, Any], files: list[dict[str, Any]], groups: Sequence[str]
    ) -> dict[str, Share]:
        tau = float(header["tau"])
        if tau != float(arg_of(header, "--tau")):
            raise SystemExit(f"{label}: the header's tau differs from its --tau")
        per_file: list[list[Offer]] = []
        words = 0
        for r in files:
            offers: list[Offer] = []
            for t in r["turns"]:
                words += prepare_turn(t, guarded=True).final_words
                offers += offers_of_turn(t, tau)
            per_file.append(offers)
        if (len(per_file), words) != (FILES, WORDS):
            raise SystemExit(
                f"{label}: {len(per_file)} files, {words} words; want {FILES}, {WORDS}"
            )
        every = [o for f in per_file for o in f]
        say(
            f"=== {label}: {arg_of(header, '--config')} seed {arg_of(header, '--seed')} tau {tau}; "
            f"VALIDITY {len(per_file)} files, {words} scored final words; {len(every)} offers "
            "with a scored fresh word, each record checked (twl.step2_stops)"
        )
        say(
            f"STOPS {label}: "
            + "; ".join(
                f"{s} {sum(o.stop == s for o in every)}" for s in ("START", "INSIDE", "END")
            )
            + "; CONTEXTS: "
            + "; ".join(f"{c} {sum(o.context == c for o in every)}" for c in ("OPEN", *AFTER))
        )
        rng = random.Random(0)
        n = len(per_file)
        dr = [[rng.randrange(n) for _ in range(n)] for _ in range(BOOT)]
        ix = list(range(n))
        out: dict[str, Share] = {}

        def ci(what: str, vals: Sequence[float]) -> tuple[float, float]:
            empty = sum(1 for x in vals if x != x)
            if empty:
                say(
                    f"NOTE {label}: {empty} of {len(vals)} resamples leave {what} empty; "
                    "its CI is over the rest"
                )
            return pct(vals)

        for g in groups:
            sel = group_of(g)
            den = [sum(1 for o in f if sel(o)) for f in per_file]
            start = [sum(1 for o in f if sel(o) and o.stop == "START") for f in per_file]
            below = [sum(1 for o in f if sel(o) and o.below_first) for f in per_file]
            surv = [sum(1 for o in f if sel(o) and o.survives_first) for f in per_file]
            notok = sum(1 for f in per_file for o in f if sel(o) and not o.token_first)
            ps = [o.p_first for f in per_file for o in f if sel(o)]
            st_b = [ratio(start, den, d) for d in dr]
            be_b = [ratio(below, den, d) for d in dr]
            sv_b = [ratio(surv, den, d) for d in dr]
            st, be, sv = ratio(start, den, ix), ratio(below, den, ix), ratio(surv, den, ix)
            s_lo, s_hi = ci(f"{g}", st_b)
            b_lo, b_hi = pct(be_b)  # the same denominators: the NOTE above covers them
            v_lo, v_hi = pct(sv_b)
            med = statistics.median(ps) if ps else float("nan")
            say(
                f"GROUP {label} {g}: {sum(den)} offers; START {sum(start)} = {st:.4f} "
                f"[{s_lo:.4f}, {s_hi:.4f}]; first word: median P {med:.4f} (no CI); "
                f"P < tau {sum(below)} = {be:.4f} [{b_lo:.4f}, {b_hi:.4f}]; survives "
                f"{sum(surv)} = {sv:.4f} [{v_lo:.4f}, {v_hi:.4f}] (no number token, so not "
                f"surviving: {notok})"
            )
            out[g] = Share(st_b, st, sum(den))
            out[g + " /below"] = Share(be_b, be, sum(den))
        return out

    def diff(
        label: str, what: str, a: Share, b: Share, a_name: str, b_name: str
    ) -> tuple[float, float, float, int]:
        d = [x - y for x, y in zip(a.boot, b.boot, strict=True)]
        empty = sum(1 for x in d if x != x)
        if empty:
            say(
                f"NOTE {label}: {empty} of {len(d)} resamples leave a group empty; "
                "the CI is over the rest"
            )
        lo, hi = pct(d)
        pt = a.point - b.point
        say(f"DIFFERENCE {label} {what} {a_name} - {b_name}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        return pt, lo, hi, empty

    r1 = ("FROM THE OPEN", "AFTER A COMMIT")
    g2 = read_run("run (2)", h2, files2, (*r1, *AFTER, POOLED))
    diff("run (2)", "R1 START share", g2["AFTER A COMMIT"], g2["FROM THE OPEN"],
         "AFTER A COMMIT", "FROM THE OPEN")  # fmt: skip
    st = diff("run (2)", "R2 START share", g2[POOLED], g2["SEGMENT END"], POOLED, "SEGMENT END")
    be = diff(
        "run (2)", "R2 first word's share with P < tau",
        g2[POOLED + " /below"], g2["SEGMENT END /below"], POOLED, "SEGMENT END",
    )  # fmt: skip
    empty = [g for g in (POOLED, "SEGMENT END") if g2[g].n == 0]
    # strict, for the reading only: any resample that leaves either group empty
    # makes its CI one that cannot be computed on every draw
    if empty or st[3] or be[3] or any(x != x for x in (*st[:3], *be[:3])):
        reading = "NOT SUPPORTED (not computable)"
    elif st[1] > 0 and be[1] > 0:
        reading = "SUPPORTED (descriptively)"
    else:
        reading = "NOT SUPPORTED"
    say(
        f"READING (item 9, Ali's hypothesis, on run (2)): {reading} (START difference CI "
        f"entirely above 0: {st[1] > 0}; P < tau difference CI entirely above 0: {be[1] > 0}"
        + (f"; empty: {', '.join(empty)}" if empty else "")
        + (f"; resamples leaving a group empty: {max(st[3], be[3])}" if st[3] or be[3] else "")
        + "). Not a verdict; the association is not the cause; NOT SUPPORTED says the "
        "hypothesis is not shown on this stream, not that it is false"
    )
    gs = read_run("the segment loop (beside)", hs, files_s, r1)
    diff("the segment loop (beside)", "R1 START share", gs["AFTER A COMMIT"],
         gs["FROM THE OPEN"], "AFTER A COMMIT", "FROM THE OPEN")  # fmt: skip
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
