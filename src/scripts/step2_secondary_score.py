"""Step 2's secondary prediction (c), scored: the secondary model against the primary.

PRE-REGISTRATION 2026-11-24 (345efad), SECONDARY (c), with Ali's words
(decision 7, Q4), and the AMENDMENT of 2026-11-24 (c0c9c32), item 3: "(c),
the secondary against the primary: both at their own tau*, the difference in
correct-committed fraction (secondary - primary), paired per file over the
selection set, with a percentile bootstrap CI over files (2000 resamples,
seed 0). MET if the CI lies entirely at or below 0.05; a CI straddling 0.05
is NOT MET; above it, the primary's generalization is the finding, reported
as such (345efad). The primary is the one scored in e207b70 (L63_R200_text,
seed 2, tau* 0.64). If the closed loop sends the primary to the retrain
branch, (c) is reported against the retrained primary as well." Committed
before the secondary's first fit (item 5).

EACH MODEL as its own scoring log (step2_score.py) chose it: the
configuration with the lowest mean held-out log-loss, the median seed, the
guarded tau*; its predictions on the selection set, counted guarded by
step2_score's own functions (load, counts_file, model_maker), on step 1's
stream (off-policy, as the secondaries' verdicts are: the AMENDMENT of
2026-11-24 on the retrained model, D (b)). The verdict: twl.step2_rules.
c_verdict. The first --primary is the registered partner and must be named
"first" and be e207b70's choice (L63_R200_text, seed 2, tau* 0.64), or the
scorer stops; each later one is reported as well. If a model's scoring log
has no tau* (it commits nothing at LA-2's rate), (c) is not scorable and
the scorer stops saying so, for Ali.

VALIDITY, before any secondary output is read: the secondary's manifest
names the three primary corpora and mqa-multi_step, and its multi_step
files are none of the sealed 80 (twl.seal, on the label files). Then: the
selection set's label files equal the anchors log's (sha256), 574 files
and 15220 scored final words; each scoring log's build is clean and its
--models is the directory given; each model's correct-committed words at
its tau* reproduce its scoring log's PRIMARY guarded line.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_secondary_score.py
        --secondary results/raw/step2/secondary SCORE_LOG
        --primary first results/raw/step2/primary
                  results/raw/run_logs/20261124_step2_score/step2_score.log
        --primary retrained results/raw/step2/retrained
                  results/raw/run_logs/20261125_step2_score_retrained/step2_score.log
        --anchors results/raw/run_logs/20261124_step2_anchors/anchors.log [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from scripts.step2_score import (
    MULTI,
    SEAL_FILE,
    SEAL_PREFIX,
    SEL,
    SEL_FILES,
    SEL_WORDS,
    SELECTION_FROM,
    TRAIN,
    counts_file,
    draws,
    load,
    model_maker,
    pct,
    sha,
)
from twl.seal import check_unsealed, sealed_indices
from twl.step2_rules import C_BAR, c_verdict

# (c)'s registered partner: the primary as e207b70 scored it (c0c9c32 item 3)
REGISTERED = ("first", "L63_R200_text", 2, 0.64)


def choice(log: Path, models: Path) -> tuple[str, int, float, int]:
    """(configuration, median seed, tau*, correct-committed words) from a scoring log."""
    text = log.read_text()
    if re.search(r"^PRIMARY guarded \(scored\): tau\* None", text, re.M):
        raise SystemExit(
            f"(c) not scorable: {log} has no tau* (the model commits nothing at LA-2's rate); "
            "this goes to Ali"
        )
    head = re.search(r"^# step2_score\.py build (\S+); .*--models (\S+)", text, re.M)
    chosen = re.search(r"-> (\S+) \(lowest mean held-out log-loss", text)
    seed = re.search(r"^MEDIAN SEED (\d+);", text, re.M)
    prim = re.search(
        r"^PRIMARY guarded \(scored\): tau\* ([0-9.]+); .*?model correct-committed (\d+),",
        text,
        re.M,
    )
    if not (head and chosen and seed and prim):
        raise SystemExit(f"{log}: no build, --models, configuration, median seed or tau*")
    if head.group(1).endswith("-dirty"):
        raise SystemExit(f"{log}: scored on a dirty build {head.group(1)}")
    if Path(head.group(2)) != models:
        raise SystemExit(f"{log}: scored {head.group(2)}, not {models}")
    return chosen.group(1), int(seed.group(1)), float(prim.group(1)), int(prim.group(2))


def sel_preds(models: Path, config: str, seed: int) -> dict[str, dict[tuple[int, int, int], float]]:
    """The selection set's P per file, keyed (turn, hyp_index, word_index), as step2_score."""
    d: dict[str, dict[tuple[int, int, int], float]] = defaultdict(dict)
    for r in pq.read_table(models / "preds" / f"{config}_s{seed}.parquet").to_pylist():
        if r["corpus"] == SEL:
            d[str(r["utt"])][(int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))] = float(
                r["p"]
            )
    return d


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--secondary", nargs=2, required=True, metavar=("MODELS", "SCORE_LOG"))
    p.add_argument(
        "--primary", nargs=3, action="append", required=True, metavar=("NAME", "MODELS", "LOG")
    )
    p.add_argument("--anchors", type=Path, required=True, help="step2_anchors.py's log")
    p.add_argument("--labels", type=Path, default=Path("results/raw/commit_labels/num"))
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
    lines = [f"# step2_secondary_score.py build {build}; {' '.join(sys.argv[1:])}"]

    def say(line: str) -> None:
        lines.append(line)
        print(line, flush=True)

    # ---- VALIDITY: the seal, before any secondary output is read
    sec_dir, sec_log = Path(args.secondary[0]), Path(args.secondary[1])
    manifest = json.loads((sec_dir / "manifest.json").read_text())
    if manifest.get("corpora") != [*TRAIN, MULTI] or manifest.get("secondary") is not True:
        raise SystemExit(f"{sec_dir}: not the secondary's directory (its manifest's corpora)")
    sealed = sealed_indices(SEAL_FILE.read_text())
    n = {}
    for suffix in (".parquet", ".utterances.parquet"):
        f = args.labels / f"{MULTI}{suffix}"
        if sha(f) != manifest["sha256"].get(str(f)):
            raise SystemExit(f"{f}: sha256 not the secondary's manifest's")
        t = pq.read_table(f, columns=["utt"])
        n[suffix] = check_unsealed(set(map(str, t.column("utt").to_pylist())), sealed, SEAL_PREFIX)
    say(
        f"VALIDITY seal: the secondary's {MULTI} files ({n['.utterances.parquet']}; the label "
        f"files as its manifest hashed them) are none of the sealed {len(sealed)} ({SEAL_FILE})"
    )

    anchors = args.anchors.read_text().splitlines()
    for path in (args.labels / f"{SEL}.parquet", args.labels / f"{SEL}.utterances.parquet"):
        if f"# sha256 {sha(path)} {path}" not in anchors:
            raise SystemExit(f"{path}: sha256 not the anchors log's")
    files, _tail_guard = load(args.labels, SEL, SELECTION_FROM)
    words = [sum(prep.final_words for prep in f.prep[True]) for f in files]
    if (len(files), sum(words)) != (SEL_FILES, SEL_WORDS):
        raise SystemExit(f"selection set {len(files)} files, {sum(words)} words")
    say(
        f"VALIDITY selection set: {len(files)} files, {sum(words)} scored final words; its label "
        f"files equal {args.anchors}'s"
    )

    models = [("secondary", sec_dir, sec_log)] + [
        (name, Path(d), Path(lg)) for name, d, lg in args.primary
    ]
    cw: dict[str, list[int]] = {}
    dr = draws(len(files))
    for name, d, lg in models:
        config, seed, tau, logged = choice(lg, d)
        if name == models[1][0] and (name, config, seed, tau) != REGISTERED:
            raise SystemExit(
                f"the first --primary is the registered partner {REGISTERED} (e207b70), not "
                f"{(name, config, seed, tau)}"
            )
        pr = sel_preds(d, config, seed)
        per = [counts_file(f, True, model_maker(pr.get(f.utt, {}), tau)) for f in files]
        if [c.words for c in per] != words:
            raise SystemExit(f"{name}: per-file words differ from the selection set's")
        cw[name] = [c.correct_words for c in per]
        if sum(cw[name]) != logged:
            raise SystemExit(f"{name}: correct-committed {sum(cw[name])}, its log {logged}")
        lo, hi = pct([sum(cw[name][i] for i in x) / sum(words[i] for i in x) for x in dr])
        say(
            f"MODEL {name} ({d}): {config}, seed {seed}, tau* {tau}, as {lg} chose it; "
            f"correct-committed {sum(cw[name]) / sum(words):.4f} [{lo:.4f}, {hi:.4f}] "
            f"({sum(cw[name])} of {sum(words)} words; reproduces its log's PRIMARY line)"
        )

    sec = cw["secondary"]
    for k, (name, _d, _lg) in enumerate(models[1:]):
        prim = cw[name]
        point = (sum(sec) - sum(prim)) / sum(words)
        lo, hi = pct(
            [
                (sum(sec[i] for i in x) - sum(prim[i] for i in x)) / sum(words[i] for i in x)
                for x in dr
            ]
        )
        role = "registered (345efad; c0c9c32 item 3)" if k == 0 else "reported as well (item 3)"
        say(
            f"(c) vs {name}, {role}: correct-committed fraction, secondary - {name}, paired by "
            f"file: {point:+.4f} [{lo:+.4f}, {hi:+.4f}] (bar {C_BAR}) -> {c_verdict(lo, hi)}"
        )
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
