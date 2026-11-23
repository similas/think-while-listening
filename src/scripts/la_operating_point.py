"""LocalAgreement-2's operating point on step 1's labels, per corpus (twl.la_baseline).

Per corpus, under the training label (label_num), each ratio pooled over word
instances with a percentile bootstrap CI over the files whose denominator is
non-zero (2000 resamples, seed 0), and that file count:
- LA2: committed and correct-committed fraction of the finals' scored words,
  precision, and declined survival split into later-committed and
  never-committed instances (definitions in twl.la_baseline). LA-2 PER WORD:
  the replay's departure; the deployed LA-2 commits per segment.
- RUNAWAY: hypotheses stuck on one token (at least 20 scored words, one key
  50 % or more of them), and their share of rows; every LA2
  and NUMBERS line is printed with them ("all") and without ("no-runaway").
- NUMBERS: survival of words in number units (the rows' num_unit column,
  twl.survival_labels.is_number_unit: a one-word "one", "ones", "oh" or "o"
  is not counted, being a pronoun or interjection as often as a number)
  against the other words, and their paired difference over the same files.
- LA_CHECK: LA-2 re-simulated on its own recorded hypotheses, matched by
  position, against the logged la2_commit.
LA-3 is not reported: it cannot be read off LA-2's stream (twl.la_baseline).

The replay's tail guard and agreement order are read from each label file's
provenance, not assumed. The label files' label_num must be the current
version (twl.survival_labels.LABEL_VERSION, from relabel_num.py or a
replay under this code): the number-normalized scorer on number units.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    la_operating_point.py [--labels results/raw/commit_labels/num] [--log FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from twl.la_baseline import (
    LABEL,
    Counts,
    la2_counts,
    runaway_hyps,
    scored_words,
    simulated_commits,
)
from twl.survival_labels import LABEL_VERSION

CORPORA = (
    "mqa-short_digit",
    "mqa-long_digit",
    "mqa-single_step",
    "mqa-multi_step",
    "librispeech-test-clean",
    "librispeech-test-other",
    "peoples-speech",
)
ROW_COLS = [
    "utt", "turn", "hyp_index", "word_index", "start_s", "end_s", "key", "text",
    "la2_commit", LABEL, "num_unit",
]  # fmt: skip
BOOT = 2000
NORMALIZER = importlib.metadata.version("whisper-normalizer")


def boot(num: list[int], den: list[int], seed: int = 0) -> str:
    """'a/b = r [lo, hi] (files n)': pooled ratio, bootstrap over units with den > 0."""
    units = [(a, b) for a, b in zip(num, den, strict=True) if b > 0]
    if not units:
        return "n/a (files 0)"
    a_sum, b_sum = sum(a for a, _ in units), sum(b for _, b in units)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOT):
        pick = [units[rng.randrange(len(units))] for _ in units]
        draws.append(sum(a for a, _ in pick) / sum(b for _, b in pick))
    draws.sort()
    lo, hi = draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]
    return f"{a_sum}/{b_sum} = {a_sum / b_sum:.4f} [{lo:.4f}, {hi:.4f}] (files {len(units)})"


def boot_diff(a_num: list[int], a_den: list[int], b_num: list[int], b_den: list[int]) -> str:
    """Paired difference of two pooled ratios, a - b, bootstrapped over the same files
    (those with any denominator); a resample with an empty denominator is drawn again."""
    units = [u for u in zip(a_num, a_den, b_num, b_den, strict=True) if u[1] + u[3] > 0]
    if not units or sum(u[1] for u in units) == 0 or sum(u[3] for u in units) == 0:
        return "n/a (files 0)"

    def diff(us: list[tuple[int, int, int, int]]) -> float:
        return sum(u[0] for u in us) / sum(u[1] for u in us) - sum(u[2] for u in us) / sum(
            u[3] for u in us
        )

    rng = random.Random(0)
    draws: list[float] = []
    while len(draws) < BOOT:
        pick = [units[rng.randrange(len(units))] for _ in units]
        if sum(u[1] for u in pick) and sum(u[3] for u in pick):
            draws.append(diff(pick))
    draws.sort()
    lo, hi = draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]
    return f"{diff(units):+.4f} [{lo:+.4f}, {hi:+.4f}] (files {len(units)})"


def la2_line(corpus: str, tag: str, per_file: list[Counts], order: int = 2) -> str:
    def col(attr: str) -> list[int]:
        return [int(getattr(c, attr)) for c in per_file]

    com, fin, surv = col("committed"), col("final_words"), col("committed_survive")
    dec, dsurv = col("declined"), col("declined_survive")
    later, lsurv = col("declined_later"), col("declined_later_survive")
    never = [d - lt for d, lt in zip(dec, later, strict=True)]
    nsurv = [d - lt for d, lt in zip(dsurv, lsurv, strict=True)]
    return (
        f"LA{order} {corpus} [{tag}]: committed {boot(com, fin)}; correct committed "
        f"{boot(surv, fin)}; precision {boot(surv, com)}; declined survival "
        f"{boot(dsurv, dec)}; of them later committed {boot(lsurv, later)}; never "
        f"committed {boot(nsurv, never)}"
    )


def provenance(path: Path) -> dict[str, object]:
    meta = pq.read_schema(path).metadata or {}
    prov: dict[str, object] = json.loads(meta[b"twl_provenance"])
    return prov


def corpus_report(labels: Path, corpus: str) -> list[str]:
    rows_path = labels / f"{corpus}.parquet"
    prov = provenance(rows_path)
    relabel = prov.get("relabel")
    version = (relabel if isinstance(relabel, dict) else prov.get("settings", {})).get(
        "label_num_version"
    )
    if version != LABEL_VERSION:
        raise SystemExit(f"{rows_path}: label_num version {version!r}, not {LABEL_VERSION!r}")
    recorded = (
        relabel.get("whisper_normalizer")
        if isinstance(relabel, dict)
        else prov.get("settings", {}).get("packages", {}).get("whisper-normalizer")
    )
    if recorded != NORMALIZER:  # finals' scored words are recomputed with the installed one
        raise SystemExit(
            f"{rows_path}: labelled under whisper-normalizer {recorded}, not {NORMALIZER}"
        )
    replay = prov["settings"]["replay"]  # type: ignore[index]
    tail_guard, order = float(replay["tail_guard_s"]), int(replay["agreement_n"])
    if order not in (2, 3):
        raise SystemExit(f"{corpus}: the replay's committer is LA-{order}, not LA-2")
    by_utt: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in pq.read_table(rows_path, columns=ROW_COLS).to_pylist():
        by_utt[str(r["utt"])].append(r)
    utts = pq.read_table(
        labels / f"{corpus}.utterances.parquet", columns=["utt", "turns"]
    ).to_pylist()
    out: list[str] = []
    per: dict[str, list[Counts]] = {"all": [], "no-runaway": []}
    nums: dict[str, dict[str, list[list[int]]]] = {
        t: {"num": [[], []], "other": [[], []]} for t in per
    }
    runaway_rows = runaway_n = hyps_n = rows_n = 0
    runaway_files: Counter[str] = Counter()
    reproduced = checked = 0
    for u in utts:
        uid = str(u["utt"])
        rows = by_utt.get(uid, [])
        turns = u["turns"]
        finals = sum(scored_words([str(w["text"]) for w in t["final_words"]]) for t in turns)
        stuck = runaway_hyps(rows)
        clean = [r for r in rows if (int(r["turn"]), int(r["hyp_index"])) not in stuck]
        hyps_n += len({(r["turn"], r["hyp_index"]) for r in rows})
        rows_n += len(rows)
        runaway_n += len(stuck)
        runaway_rows += len(rows) - len(clean)
        if stuck:
            runaway_files[uid] += len(rows) - len(clean)
        per["all"].append(la2_counts(rows, finals))
        per["no-runaway"].append(la2_counts(clean, finals))
        for tag, rs in (("all", rows), ("no-runaway", clean)):
            for kind in ("num", "other"):
                sel = [
                    r for r in rs
                    if int(r[LABEL]) >= 0 and int(r["num_unit"]) == (kind == "num")
                ]  # fmt: skip
                nums[tag][kind][0].append(sum(int(r[LABEL]) for r in sel))
                nums[tag][kind][1].append(len(sel))
        sim: set[tuple[int, int, int]] = set()
        for k, t in enumerate(turns):
            sim |= {(k, h, i) for h, i in simulated_commits(t["hyps"], order, tail_guard)}
        for r in rows:
            key = (int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))
            checked += 1
            reproduced += int((key in sim) == (int(r["la2_commit"]) == 1))
    for tag, counts in per.items():
        out.append(la2_line(corpus, tag, counts, order))
    for tag in per:
        n, o = nums[tag]["num"], nums[tag]["other"]
        out.append(
            f"NUMBERS {corpus} [{tag}]: number words survive {boot(n[0], n[1])}; "
            f"other words {boot(o[0], o[1])}; paired difference (number - other) "
            f"{boot_diff(n[0], n[1], o[0], o[1])}"
        )
    top = ", ".join(f"{u} ({k} rows)" for u, k in runaway_files.most_common(3))
    out.append(
        f"RUNAWAY {corpus}: hypotheses {runaway_n} of {hyps_n}; rows {runaway_rows} of "
        f"{rows_n}; files {len(runaway_files)}; largest: {top or 'none'}"
    )
    out.append(
        f"LA_CHECK {corpus}: LA-{order} re-simulated (tail guard {tail_guard}) reproduces "
        f"la2_commit on {reproduced}/{checked} rows, matched by position"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--labels", type=Path, default=Path("results/raw/commit_labels/num"))
    p.add_argument("--log", type=Path, default=None, help="also write the output here")
    p.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="a label file's stem, e.g. mqa-single_step.la3 (repeatable; default the seven)",
    )
    args = p.parse_args()
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if args.log and commit.endswith("-dirty"):
        raise SystemExit(f"build {commit} is dirty: a logged run needs a clean tree")
    lines = [
        f"# la_operating_point.py build {commit}; whisper-normalizer {NORMALIZER}; "
        f"{' '.join(sys.argv[1:])}"
    ]
    corpora = tuple(args.corpus) if args.corpus else CORPORA
    for corpus in corpora:
        for suffix in (".parquet", ".utterances.parquet"):
            f = args.labels / f"{corpus}{suffix}"
            lines.append(f"# sha256 {hashlib.sha256(f.read_bytes()).hexdigest()} {f}")
    for corpus in corpora:
        lines += corpus_report(args.labels, corpus)
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
