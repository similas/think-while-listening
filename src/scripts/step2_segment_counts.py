"""Step 2: how many decoder segments the offered hypotheses hold, logged.

Ali (2026-11-27): "log the per-hypothesis segment count and the
last-segment fresh-word fraction in the replay output so they're recorded,
not read by hand." Descriptive; no verdict.

For a run whose hypothesis words carry their segment (the segment replay,
step2_segment_replay.py; a loop under the segment rule, step2_closed_loop.py
--segments), over the offered hypotheses that have a fresh word
(twl.step2_segments.segment_counts), twice: all of them (the runaway
hypotheses included, as the count of 82d9fea was made), and those the
runaway guard keeps (the ones the guarded count offers to a committer):
- SEGMENTS: how many hypotheses hold 1, 2, 3 or 4 and more segments among
  their fresh words, and the share with one, with a 95 % percentile
  bootstrap CI over files (2000 resamples, seed 0);
- LAST SEGMENT: of their fresh words, how many lie in their hypothesis's
  last segment, the share with its CI; the rest is what item 2 of the
  AMENDMENT of 2026-11-26 can commit at that offer (D1 (i)).

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_segment_counts.py --run results/raw/step2/segment_replay/segment_replay.jsonl
        [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from scripts.la_operating_point import BOOT
from twl.step2_segments import segment_counts


def pct(vals: list[float]) -> tuple[float, float]:
    v = sorted(x for x in vals if x == x)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
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
    lines = [f"# step2_segment_counts.py build {build}; {' '.join(sys.argv[1:])}"]

    def say(line: str) -> None:
        lines.append(line)
        print(line, flush=True)

    recs = [json.loads(x) for x in args.run.read_text().splitlines() if x.strip()]
    header, files = recs[0]["header"], recs[1:]
    if header.get("limit"):
        raise SystemExit(f"{args.run}: a smoke run (limit {header['limit']})")
    if header.get("segments") is False:
        raise SystemExit(f"{args.run}: its words carry no segment (a plain-rule loop)")
    per_file = [[c for t in r["turns"] for c in segment_counts(t["hyps"])] for r in files]
    say(
        f"RUN {args.run}: build {str(header['build'])[:7]}; {len(files)} files; "
        f"segments recorded: {'yes' if header.get('segments', True) else 'no'}"
    )
    rng = random.Random(0)
    dr = [[rng.randrange(len(files)) for _ in files] for _ in range(BOOT)]
    for label, keep in (("all offered", lambda c: True), ("the guard's kept", lambda c: not c[3])):
        cs = [[c for c in f if keep(c)] for f in per_file]
        n = [len(f) for f in cs]
        one = [sum(1 for c in f if c[0] == 1) for f in cs]
        words = [sum(c[1] for c in f) for f in cs]
        last = [sum(c[2] for c in f) for f in cs]
        dist = {k: sum(1 for f in cs for c in f if min(c[0], 4) == k) for k in (1, 2, 3, 4)}

        def share(num: list[int], den: list[int]) -> str:
            pt = sum(num) / sum(den)
            lo, hi = pct([sum(num[i] for i in d) / max(1, sum(den[i] for i in d)) for d in dr])
            return f"{pt:.4f} [{lo:.4f}, {hi:.4f}]"

        say(
            f"SEGMENTS ({label}; hypotheses with a fresh word): {sum(n)}; holding 1 segment "
            f"{dist[1]}, 2 {dist[2]}, 3 {dist[3]}, 4 or more {dist[4]}; the share with one "
            f"{share(one, n)}"
        )
        say(
            f"LAST SEGMENT ({label}): {sum(last)} of {sum(words)} fresh words lie in their "
            f"hypothesis's last segment, {share(last, words)}; before it "
            f"{sum(words) - sum(last)}"
        )
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
