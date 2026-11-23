"""Recompute step 1's training label with the number-normalized scorer (Ali, 2026-11-19).

Step 1 wrote ``label_num`` with an interim per-word number normalizer. This
recomputes it, and ``match_offset_s`` with it, under twl.survival_labels'
number units (twl.wer.normalize_num, Whisper's English normalizer), from the
hypothesis and final words every utterance record keeps. Nothing is
re-decoded.

Each row is found again by POSITION: the turn, the offered hypothesis
(frozen ones are not counted, as in the replay), and the word's index among
that hypothesis's fresh words. The recomputation refuses a row whose text is
not the word at that position.

Writes, per corpus, under --out (default results/raw/commit_labels/num/):
- <corpus>.parquet: step 1's rows with label_num and match_offset_s
  recomputed, a num_unit column (the word is in a number unit), and the
  step-1 values kept as label_num_interim and match_offset_interim_s; the
  provenance gains a "relabel" entry (build, invocation, normalizer
  version, both inputs' sha256);
- <corpus>.utterances.parquet: a byte copy of step 1's, checked by sha256.
Step 1's own files are only read (their sha256 is in the LA-2 operating
point's log): --out may not be --labels, and an input already relabelled is
refused.

The ROW SET is step 1's. Under the new is_scored a few words would be rows
or not where step 1's was different; ROWSET counts them.

Prints per corpus:
- RELABEL: rows; label_num = 1 before and after; rows whose label changed,
  each way; rows in a number unit, and of them in a unit of two or more
  words; unit-edge zeros: rows labelled 0 in a number unit whose own word
  survives as a word (label_exact 1): its unit differs from the final's
  (a part not yet heard, a part misheard, a sign added or missing); rows
  this code does not score (is_scored), set to -1 as rows() would leave
  them out;
- UNITS: turn finals whose units' tokens equal the scorer's tokens of the
  same words joined (twl.survival_labels.join_words) and followed by more
  speech; how many differ from the words normalized alone (Whisper keeps a
  "." at the very end of a string ending in "..."); and against the turn's
  stored final string, so the word-level label can be compared with the
  scorer. Step 1 stored that string as the words joined with spaces, which
  splits a decimal ("1 .77"); commit_labels.py joins them with join_words
  from 2026-11-22 on. The same check on every offered hypothesis's fresh
  words.
- ROWSET: fresh words the new is_scored scores that have no row, and rows
  whose word it no longer scores (labelled -1 above).

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    relabel_num.py [--labels results/raw/commit_labels] [--out DIR] [--log FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from twl.commit import Word
from twl.survival_labels import (
    LABEL_VERSION,
    NUM_UNIT_DESCRIPTION,
    _tokens,
    is_scored,
    join_words,
    num_labels,
    num_units,
)
from twl.wer import normalize_num

CORPORA = (
    "mqa-short_digit",
    "mqa-long_digit",
    "mqa-single_step",
    "mqa-multi_step",
    "librispeech-test-clean",
    "librispeech-test-other",
    "peoples-speech",
)
FRESH_EPS = 1e-9  # twl.survival_labels.fresh_words


def words(raw: list[dict[str, object]]) -> list[Word]:
    return [Word(str(w["text"]), float(str(w["start_s"])), float(str(w["end_s"]))) for w in raw]


def in_context(texts: list[str]) -> list[str]:
    """The scorer's tokens of these words followed by more speech (a sentinel
    word, dropped): what num_units composes. Alone, a string ending in "..."
    keeps a "." token that any following word removes."""
    return list(_tokens(texts, 0, len(texts)))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relabel(labels: Path, out: Path, corpus: str, build: str, argv: str) -> list[str]:
    src_rows = labels / f"{corpus}.parquet"
    src_utts = labels / f"{corpus}.utterances.parquet"
    table = pq.read_table(src_rows)
    meta = dict(table.schema.metadata or {})
    prov = json.loads(meta[b"twl_provenance"])
    if "relabel" in prov or prov.get("settings", {}).get("label_num_version") == LABEL_VERSION:
        raise SystemExit(f"{src_rows}: label_num is already {LABEL_VERSION}")
    utts = pq.read_table(src_utts, columns=["utt", "turns"]).to_pylist()
    new: dict[tuple[str, int, int, int], tuple[int, float, int, str, int]] = {}
    scored_words: set[tuple[str, int, int, int]] = set()
    finals = same_joined = same_string = hyps = hyps_same = alone = 0
    for u in utts:
        uid = str(u["utt"])
        for k, t in enumerate(u["turns"]):
            final = words(t["final_words"])
            texts = [w.text for w in final]
            toks = [x for unit in num_units(texts) for x in unit[1]]
            finals += 1
            same_joined += toks == in_context(texts)
            alone += toks != normalize_num(join_words(texts))
            same_string += toks == normalize_num(str(t["final"]))
            h = 0
            for hyp in t["hyps"]:
                if hyp["frozen"]:
                    continue
                h += 1
                ce = float(hyp["committed_end_s"])
                fresh = [w for w in words(hyp["words"]) if w.start_s >= ce - FRESH_EPS]
                ftexts = [w.text for w in fresh]
                units = num_units(ftexts)
                size = {i: len(un[0]) for un in units for i in un[0]}
                hyps += 1
                hyps_same += [x for un in units for x in un[1]] == in_context(ftexts)
                for i, (lab, off, num) in enumerate(num_labels(fresh, final)):
                    new[(uid, k, h, i)] = (lab, off, num, fresh[i].text, size[i])
                    if is_scored(fresh[i]):
                        scored_words.add((uid, k, h, i))
    rows = table.select(
        ["utt", "turn", "hyp_index", "word_index", "text", "label_exact"]
    ).to_pylist()
    dropped = 0
    lab_new: list[int] = []
    off_new: list[float] = []
    num_new: list[int] = []
    keys = set()
    for r in rows:
        key = (str(r["utt"]), int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))
        keys.add(key)
        if key not in new:
            raise SystemExit(f"{corpus}: row {key} has no word at that position")
        lab, off, num, text, _ = new[key]
        if text != r["text"]:
            raise SystemExit(f"{corpus}: row {key} is {r['text']!r}, the word there is {text!r}")
        if key not in scored_words:  # not a row under this code: unscored, as rows() writes
            lab, off = -1, math.nan
            dropped += 1
        lab_new.append(lab)
        off_new.append(off)
        num_new.append(num)
    old = table.column("label_num").to_pylist()
    idx = table.schema.get_field_index("label_num")
    oidx = table.schema.get_field_index("match_offset_s")
    table = table.append_column("label_num_interim", table.column("label_num"))
    table = table.append_column("match_offset_interim_s", table.column("match_offset_s"))
    table = table.set_column(idx, "label_num", pa.array(lab_new, pa.int8()))
    table = table.set_column(oidx, "match_offset_s", pa.array(off_new, pa.float64()))
    table = table.append_column("num_unit", pa.array(num_new, pa.int8()))
    settings = prov.get("settings", {})
    if isinstance(settings.get("labels"), dict) and "label_num" in settings["labels"]:
        settings["labels"]["label_num"] = (
            "SUPERSEDED by the relabel entry; step 1's description: "
            + str(settings["labels"]["label_num"])
        )
    prov["relabel"] = {
        "script": "src/scripts/relabel_num.py",
        "build": build,
        "invocation": argv,
        "label_num": "number-normalized scorer (twl.wer.normalize_num) on number units "
        "(twl.survival_labels.num_units, in context); the step-1 column is label_num_interim",
        "whisper_normalizer": importlib.metadata.version("whisper-normalizer"),
        "label_num_version": LABEL_VERSION,
        "rows_sha256": sha256(src_rows),
        "utterances_sha256": sha256(src_utts),
    }
    if isinstance(settings, dict):
        settings.setdefault("descriptor_columns", {})["num_unit"] = NUM_UNIT_DESCRIPTION
    labels_cols = prov.get("settings", {}).get("label_columns")
    if isinstance(labels_cols, list):
        labels_cols += ["label_num_interim", "match_offset_interim_s"]
    meta[b"twl_provenance"] = json.dumps(prov).encode()
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{corpus}.parquet.tmp"
    pq.write_table(table.replace_schema_metadata(meta), tmp)
    os.replace(tmp, out / f"{corpus}.parquet")
    copy = out / f"{corpus}.utterances.parquet"
    shutil.copyfile(src_utts, copy)
    if sha256(copy) != prov["relabel"]["utterances_sha256"]:
        raise SystemExit(f"{copy}: the copy differs from {src_utts}")
    n = len(rows)
    pairs = list(zip(old, lab_new, strict=True))
    up = sum(1 for a, b in pairs if a != 1 and b == 1)
    down = sum(1 for a, b in pairs if a == 1 and b != 1)
    to_unscored = sum(1 for a, b in pairs if a >= 0 and b < 0)
    from_unscored = sum(1 for a, b in pairs if a < 0 and b >= 0)
    ones_old = sum(1 for a in old if a == 1)
    ones_new = sum(1 for b in lab_new if b == 1)
    scored_old = sum(1 for a in old if a >= 0)
    scored_new = sum(1 for b in lab_new if b >= 0)
    in_num = sum(num_new)
    multi = sum(
        1
        for r in rows
        if (v := new[(str(r["utt"]), int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))])[2]
        and v[4] > 1
    )
    edge = sum(
        1
        for r, lab, num in zip(rows, lab_new, num_new, strict=True)
        if num and lab == 0 and int(r["label_exact"]) == 1
    )
    return [
        f"RELABEL {corpus}: rows {n}; scored {scored_old} -> {scored_new}; label_num = 1 "
        f"{ones_old}/{scored_old} = {ones_old / max(scored_old, 1):.4f} -> {ones_new}/"
        f"{scored_new} = {ones_new / max(scored_new, 1):.4f}; changed to 1 {up}, from 1 "
        f"{down}; newly unscored {to_unscored}, newly scored {from_unscored}; rows in a "
        f"number unit {in_num}, in one of two or more words {multi}; unit-edge zeros "
        f"(0 in a number unit, its own word label_exact 1) {edge}; rows this code does not "
        f"score, set to -1: {dropped}",
        f"UNITS {corpus}: turn finals {finals}; units' tokens equal the scorer's on the "
        f"joined words, in context {same_joined} ({same_joined / max(finals, 1):.4f}) (and "
        f"differ from the words normalized alone {alone}, a string ending in "
        f'"..." keeping a final "."); on the stored '
        f'final string, which step 1 joined with spaces ("1 .77"), {same_string} '
        f"({same_string / max(finals, 1):.4f}); offered hypotheses' fresh words {hyps_same}"
        f"/{hyps} ({hyps_same / max(hyps, 1):.4f})",
        f"ROWSET {corpus}: words the new is_scored scores without a row "
        f"{len(scored_words - keys)}; rows whose word it no longer scores "
        f"{len(keys - scored_words)}",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--labels", type=Path, default=Path("results/raw/commit_labels"))
    p.add_argument("--out", type=Path, default=Path("results/raw/commit_labels/num"))
    p.add_argument("--log", type=Path, default=None)
    args = p.parse_args()
    if args.out.resolve() == args.labels.resolve():
        raise SystemExit("--out is --labels: step 1's files are never written")
    under_results = args.out.resolve().is_relative_to(Path("results").resolve())
    if under_results and args.log is None:
        raise SystemExit("a relabel under results/ needs --log")
    build = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if build.endswith("-dirty") and under_results:
        raise SystemExit(f"build {build} is dirty: a relabel under results/ needs a clean tree")
    lines = [f"# relabel_num.py build {build}; {' '.join(sys.argv[1:])}"]
    for corpus in CORPORA:
        lines += relabel(args.labels, args.out, corpus, build, " ".join(sys.argv[1:]))
        sys.stdout.write("\n".join(lines[-3:]) + "\n")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
