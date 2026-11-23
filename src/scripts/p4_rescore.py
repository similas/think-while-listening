"""P4 re-scored under the number-normalized scorer, beside the registered one (post hoc).

Ali's scorer decision (2026-11-19), item 3: "re-score P4 for pass 2 and pass
2k with the new scorer and append both beside the registered verdicts,
labelled as a post-hoc sensitivity. The registered verdicts are not
changed." The scorers run unchanged, as the claim registry runs them, once
with the registered per-item WER (twl.wer.wer) and once with
--scorer wer_num (twl.wer.wer_num); item alignment always uses the
registered wer.

- Pass 2 (pass2_score.py --p4-mean, NOTES 2026-10-06): P4 is the paired
  dWER = WER(COMMIT-WL) - WER(NOPARTIAL) per item on all four sets, bar
  <= +0.02 (+2.0 points) on every split. The multi_step and sixteen runs are
  the registry's (sources pass2, pass2dev); the digit sets' runs are the
  same passes' (NOPARTIAL 2026-10-02, COMMIT-WL 2026-10-05).
- Pass 2k (p2k_score.py, source p2k): its own P4 lines and verdicts, and
  the mean's CI.

Prints P4 lines labelled by pass, set and scorer, and one P4 SAME line per
set: whether the sign of the mean and the bar outcome are the same under
both scorers.

    p4_rescore.py [--log FILE]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from scripts.paper_sources import REPO, SOURCES

RAW = "results/raw/reactive/reactive-"
DIGIT_SETS = {
    "short_digit": (
        "results/raw/spoken_mqa/short_digit",
        ("20261002-103332-b34440", "20261002-111420-86da6c", "20261002-115514-ed6f32"),
        ("20261005-144706-f049cb", "20261005-152813-b26387", "20261005-160916-22433d"),
    ),
    "long_digit": (
        "results/raw/spoken_mqa/long_digit",
        ("20261002-103659-426fbe", "20261002-111804-8a8ad0", "20261002-115855-7562bf"),
        ("20261005-145035-8ae463", "20261005-153144-27ee84", "20261005-161247-ea3914"),
    ),
}
BAR = 0.02  # pass 2's P4 bar: dWER <= +2.0 points on every split (NOTES 2026-10-06)
SCORERS = ("wer", "wer_num")


def run(script: str, args: list[str]) -> str:
    res = subprocess.run(
        [sys.executable, str(REPO / "src/scripts" / script), *args],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout


def pass2_sets() -> dict[str, list[str]]:
    sets = {}
    for key in ("pass2dev", "pass2"):
        args = list(SOURCES[key].args)
        sets[args[args.index("--set") + 1]] = args
    for name, (set_dir, base, test) in DIGIT_SETS.items():
        sets[name] = [
            "--set", name, "--set-dir", set_dir,
            "--base", *(RAW + r for r in base), "--test", *(RAW + r for r in test),
        ]  # fmt: skip
    return sets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--log", type=Path, default=None)
    a = p.parse_args()
    build = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    ).stdout.strip()
    if a.log and build.endswith("-dirty"):
        raise SystemExit(f"build {build} is dirty: a logged run needs a clean tree")
    lines = [f"# p4_rescore.py build {build}"]
    for name, args in pass2_sets().items():
        means = {}
        for sc in SCORERS:
            out = run("pass2_score.py", [*args, "--p4-mean", "--scorer", sc])
            ln = next(x.strip() for x in out.splitlines() if "P4 dWER" in x)
            mean = float(re.search(r"mean ([+-][0-9.]+)", ln).group(1))  # type: ignore[union-attr]
            means[sc] = mean
            lines.append(f"P4 PASS2 {name} {sc}: {ln.removeprefix('[report] ')}")
        same_sign = (means["wer"] > 0) == (means["wer_num"] > 0) or means["wer"] == means["wer_num"]
        same_bar = (means["wer"] <= BAR) == (means["wer_num"] <= BAR)
        lines.append(
            f"P4 SAME pass2 {name}: sign of the mean {'same' if same_sign else 'DIFFERS'}; bar "
            f"(mean <= +{BAR}) {'same' if same_bar else 'DIFFERS'} "
            f"({'passes' if means['wer_num'] <= BAR else 'fails'} under wer_num)"
        )
    args = list(SOURCES["p2k"].args)
    names = [args[i + 1] for i, a in enumerate(args) if a == "--set"]
    verdicts: dict[str, list[str]] = {}
    for sc in SCORERS:
        out = run("p2k_score.py", [*args, "--scorer", sc])
        p4 = [x.strip() for x in out.splitlines() if "P4 dWER STOP-NOPARTIAL" in x]
        ci = [x.strip() for x in out.splitlines() if "dWER mean CI" in x]
        for name, ln, c in zip(names, p4, ci, strict=True):
            lines.append(f"P4 PASS2K {name} {sc}: {ln} | {c.removeprefix('[report] ')}")
            verdicts.setdefault(name, []).append(ln.rsplit("-> ", 1)[-1])
    for name, (reg, num) in verdicts.items():
        lines.append(
            f"P4 SAME pass2k {name}: verdict {'same' if reg == num else 'DIFFERS'} ({num})"
        )
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if a.log:
        a.log.parent.mkdir(parents=True, exist_ok=True)
        a.log.write_text(text)


if __name__ == "__main__":
    main()
