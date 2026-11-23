"""Step 1's LABELS table, regenerated on the relabelled files (NOTES 2026-11-23).

Per corpus, from results/raw/commit_labels/num (relabel_num.py's output):
- survival of each label column (label_num, its step-1 interim, label_wer,
  label_exact): rows labelled 1 / scored rows, with a percentile bootstrap
  CI over the files with a scored row (la_operating_point.boot), [all] and
  [no-runaway] (twl.la_baseline.runaway_hyps, step 1's definition);
- the turn finals' WER against the reference under the registered scorer
  (twl.wer.wer) and the number-normalized one (wer_num): errors over
  reference words pooled, bootstrap over files. A file's hypothesis is its
  turns' finals, each joined by twl.survival_labels.join_words.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    labels_table.py [--labels results/raw/commit_labels/num] [--log FILE]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from scripts.la_operating_point import CORPORA, boot
from twl.la_baseline import runaway_hyps
from twl.survival_labels import join_words
from twl.wer import normalize, normalize_num, wer, wer_num

COLUMNS = ("label_num", "label_num_interim", "label_wer", "label_exact")


def corpus_lines(labels: Path, corpus: str) -> list[str]:
    cols = ["utt", "turn", "hyp_index", "key", *COLUMNS]
    by_utt: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in pq.read_table(labels / f"{corpus}.parquet", columns=cols).to_pylist():
        by_utt[str(r["utt"])].append(r)
    out = []
    for tag in ("all", "no-runaway"):
        parts = []
        for col in COLUMNS:
            num: list[int] = []
            den: list[int] = []
            for rows in by_utt.values():
                if tag == "no-runaway":
                    stuck = runaway_hyps(rows)
                    rows = [
                        r
                        for r in rows
                        if (int(str(r["turn"])), int(str(r["hyp_index"]))) not in stuck
                    ]
                scored = [int(str(r[col])) for r in rows if int(str(r[col])) >= 0]
                num.append(sum(scored))
                den.append(len(scored))
            parts.append(f"{col} {boot(num, den)}")
        out.append(f"LABELS {corpus} [{tag}]: " + "; ".join(parts))
    utts = pq.read_table(
        labels / f"{corpus}.utterances.parquet", columns=["utt", "reference", "turns"]
    ).to_pylist()
    for name, score, tokens in (("wer", wer, normalize), ("wer_num", wer_num, normalize_num)):
        errs: list[int] = []
        refw: list[int] = []
        for u in utts:
            ref = str(u["reference"] or "")
            hyp = " ".join(
                join_words([str(w["text"]) for w in t["final_words"]]) for t in u["turns"]
            )
            n = len(tokens(ref))
            errs.append(round(score(ref, hyp) * n))
            refw.append(n)
        out.append(f"FINAL WER {corpus} {name}: {boot(errs, refw)} (errors / reference words)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
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
    lines = [f"# labels_table.py build {build}; {' '.join(sys.argv[1:])}"]
    for corpus in CORPORA:
        lines += corpus_lines(args.labels, corpus)
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
