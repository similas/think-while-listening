"""Step 2: train the primary learned committer (Model B, gradient-boosted trees).

PRE-REGISTRATION 2026-11-24 (345efad), AMENDMENT 2439df4: every choice here is
declared there, before any fit.
- Training corpora (Ali's text, the primary model): librispeech-test-clean,
  librispeech-test-other and peoples-speech label rows (results/raw/
  commit_labels/num), label_num 0 or 1, the guard's hypotheses dropped
  (decision 5: twl.runaway on the hypothesis's words, h counted over the
  offered hypotheses as twl.step2_count does).
- Held-out split (amendment item 1): ids f"{corpus}:{utt}" for every distinct
  utt in the three label-row files; held = random.Random(0).sample(sorted(ids),
  round(0.1 * n)); every configuration and seed trains on the rest.
- Configurations (amendment item 4): the 2 x 2 x 2 grid of num_leaves 15 or
  63, 200 or 600 rounds, and the word's text (text_hash % 1024, floor modulo,
  categorical) as a feature or not; learning rate 0.05, min_data_in_leaf 200,
  feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0,
  deterministic, force_col_wise, 4 threads; seeds 0, 1, 2; the last
  checkpoint (a fixed number of rounds, no early stopping).
- Predictions (amendment item 5): the held-out training files, the selection
  set (single_step items 20 and up) and the digit sets; none on multi_step.
- The retrained model (AMENDMENT 2026-11-24, "the retrained model's verdict is
  decided by one guarded closed loop", item 1), with --onpolicy DIR: the
  on-policy rows of LibriSpeech test-clean (step2_relabel.py's
  librispeech-test-clean.onpolicy.parquet and .utterances.parquet in DIR) are
  added to all the rows above, label 0/1, the guard's hypotheses dropped
  (guarded_out on the relabel's utterances file), keyed as
  f"librispeech-test-clean:{utt}". They add no file to the split: a file
  outside step 1's test-clean stops the fit. Their rows of held-out files are
  held out. Their sha256 enter the manifest with the others. Until the first
  commit in either stream the two streams coincide, so some on-policy rows
  duplicate step 1's test-clean rows (NOTES, RECORDED BEFORE ANY RETRAINED
  FIT): the DUPLICATES line counts them (twl.step2_rows.duplicates), all and
  held out, and any with the same inputs but another label, before the fit;
  they are kept, as decision 2 (b) (i) adds the rows to all of step 1's.

Writes, under --out: manifest.json; models/<config>_s<seed>.txt;
heldout/<config>_s<seed>.parquet (source, key, turn, hyp_index, word_index,
label, p; source "step1" or "onpolicy", since the two share keys);
preds/<config>_s<seed>.parquet (corpus, utt, turn, hyp_index, word_index, p).

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_train.py --out results/raw/step2/primary
                   [--labels results/raw/commit_labels/num] [--onpolicy DIR] [--log FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from twl.runaway import runaway_reason
from twl.step2_count import words_of
from twl.step2_rows import duplicates

TRAIN = ("librispeech-test-clean", "librispeech-test-other", "peoples-speech")
SEL = "mqa-single_step"
SELECTION_FROM = 20
DIGITS = ("mqa-short_digit", "mqa-long_digit")
ONPOLICY = "librispeech-test-clean"  # the relabelled corpus (Ali's decision 2 on 54fce21)
ONPOLICY_STEM = f"{ONPOLICY}.onpolicy"
LABEL = "label_num"
NUMERIC = (
    "pos_from_end_s",
    "word_dur_s",
    "logprob",
    "no_speech_prob",
    "partials_seen",
    "agreed_prev",
    "la2_commit",
    "buffer_s",
    "since_commit_s",
    "seg_avg_logprob",
    "num_unit",
)
TEXT_BUCKETS = 1024
SEEDS = (0, 1, 2)
BASE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "verbosity": -1,
}
CONFIGS = [
    {
        "name": f"L{leaves}_R{rounds}_{'text' if text else 'notext'}",
        "num_leaves": leaves,
        "rounds": rounds,
        "text": text,
    }
    for leaves in (15, 63)
    for rounds in (200, 600)
    for text in (False, True)
]
COLUMNS = ["utt", "turn", "hyp_index", "word_index", LABEL, "text_hash", *NUMERIC]


def guarded_out(labels: Path, corpus: str) -> set[tuple[str, int, int]]:
    """(utt, turn, h) of the hypotheses the runaway guard discards."""
    out = set()
    for u in pq.read_table(
        labels / f"{corpus}.utterances.parquet", columns=["utt", "turns"]
    ).to_pylist():
        for k, t in enumerate(u["turns"]):
            h = 0
            for hyp in t["hyps"]:
                if hyp["frozen"]:
                    continue
                h += 1
                if runaway_reason(words_of(hyp["words"])):
                    out.add((str(u["utt"]), k, h))
    return out


def read_rows(labels: Path, corpus: str) -> dict[str, np.ndarray]:
    t = pq.read_table(labels / f"{corpus}.parquet", columns=COLUMNS)
    return {c: t.column(c).to_numpy(zero_copy_only=False) for c in COLUMNS}


def features(r: dict[str, np.ndarray], text: bool) -> np.ndarray:
    cols = [r[c].astype(np.float64) for c in NUMERIC]
    if text:
        cols.append(np.mod(r["text_hash"].astype(np.int64), TEXT_BUCKETS).astype(np.float64))
    return np.column_stack(cols)


def take(r: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {c: v[mask] for c, v in r.items()}


def concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}


def read_onpolicy(d: Path, ids: set[str]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """The relabel's rows, kept as the others are, keyed as step 1's test-clean files.

    Stops if a row's file is not among ids (the split's files): the on-policy
    rows add no file to the split (AMENDMENT 2026-11-24, item 1).
    """
    r = read_rows(d, ONPOLICY_STEM)
    key = np.array([f"{ONPOLICY}:{u}" for u in r["utt"]], dtype=object)
    outside = set(key.tolist()) - ids
    if outside:
        raise SystemExit(
            f"on-policy rows of {len(outside)} files outside the split: {sorted(outside)[:3]}"
        )
    drop = guarded_out(d, ONPOLICY_STEM)
    keep = np.array(
        [
            lab in (0, 1) and (str(u), int(k), int(h)) not in drop
            for lab, u, k, h in zip(r[LABEL], r["utt"], r["turn"], r["hyp_index"], strict=True)
        ]
    )
    r["key"] = key
    r["source"] = np.array(["onpolicy"] * len(key), dtype=object)
    info: dict[str, object] = {
        "dir": str(d),
        "files": len(set(key.tolist())),
        "label_rows": len(keep),
        "kept": int(keep.sum()),
    }
    return take(r, keep), info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=Path("results/raw/commit_labels/num"))
    p.add_argument("--onpolicy", type=Path, default=None, help="step2_relabel.py's output")
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
    lines = [
        f"# step2_train.py build {build}; {' '.join(sys.argv[1:])}; lightgbm {lgb.__version__}"
    ]

    def say(line: str) -> None:
        lines.append(line)
        print(line, flush=True)

    sha = {}
    for c in (*TRAIN, SEL, *DIGITS):
        for suffix in (".parquet", ".utterances.parquet"):
            f = args.labels / f"{c}{suffix}"
            sha[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
            say(f"# sha256 {sha[str(f)]} {f}")
    if args.onpolicy:
        for suffix in (".parquet", ".utterances.parquet"):
            f = args.onpolicy / f"{ONPOLICY_STEM}{suffix}"
            sha[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
            say(f"# sha256 {sha[str(f)]} {f}")

    # the training rows: label 0/1, the guard's hypotheses dropped, keyed by file
    parts, ids = [], set()
    for c in TRAIN:
        r = read_rows(args.labels, c)
        ids |= {f"{c}:{u}" for u in set(r["utt"].tolist())}
        drop = guarded_out(args.labels, c)
        key = np.array([f"{c}:{u}" for u in r["utt"]], dtype=object)
        keep = np.array(
            [
                lab in (0, 1) and (str(u), int(k), int(h)) not in drop
                for lab, u, k, h in zip(r[LABEL], r["utt"], r["turn"], r["hyp_index"], strict=True)
            ]
        )
        r["key"] = key
        r["source"] = np.array(["step1"] * len(key), dtype=object)
        parts.append(take(r, keep))
        say(
            f"ROWS {c}: {len(keep)} label rows; kept {int(keep.sum())} (label 0/1, guard's dropped)"
        )
    onpolicy: dict[str, object] | None = None
    dup_masks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    if args.onpolicy:
        op, onpolicy = read_onpolicy(args.onpolicy, ids)
        dup_masks = (
            *duplicates(parts[TRAIN.index(ONPOLICY)], op, LABEL, NUMERIC),
            op["key"],
        )
        parts.append(op)
        say(
            f"ROWS {ONPOLICY_STEM}: {onpolicy['label_rows']} label rows of {onpolicy['files']} "
            f"files; kept {onpolicy['kept']} (label 0/1, guard's dropped); no file added to the "
            "split"
        )
    rows = concat(parts)
    n = len(ids)
    held = set(random.Random(0).sample(sorted(ids), round(0.1 * n)))
    is_held = np.array([k in held for k in rows["key"]])
    rows_line = (
        f"files {n}, held out {len(held)}; training rows {int((~is_held).sum())}, "
        f"held-out rows {int(is_held.sum())}"
    )
    say(f"SPLIT: {rows_line}")
    if onpolicy is not None and dup_masks is not None:
        same_pos, same, conflict, op_keys = dup_masks
        op_held = np.array([k in held for k in op_keys], dtype=bool)
        dup = {
            "kept": len(same),
            "same_position": int(same_pos.sum()),
            "identical": int(same.sum()),
            "conflicting": int(conflict.sum()),
            "kept_held_out": int(op_held.sum()),
            "identical_held_out": int((same & op_held).sum()),
        }
        onpolicy["duplicates"] = dup
        say(
            f"DUPLICATES {ONPOLICY_STEM}: of {dup['kept']} kept rows, {dup['same_position']} "
            "share a kept step-1 row's position (file, turn, hypothesis, word) and "
            f"{dup['identical']} equal it (label, text_hash, every feature bit for bit), "
            f"{dup['conflicting']} with the same inputs but another label; held out: "
            f"{dup['identical_held_out']} identical of {dup['kept_held_out']} kept. All "
            "stay, so the identical rows weigh twice (decision 2 (b) (i))"
        )
    tr, ho = take(rows, ~is_held), take(rows, is_held)

    # the evaluation rows: the selection set (items 20 and up) and the digit sets
    evals = []
    for c in (SEL, *DIGITS):
        r = read_rows(args.labels, c)
        item = np.array([int(str(u).split(":")[-1]) for u in r["utt"]])
        r = take(r, item >= (SELECTION_FROM if c == SEL else 0))
        r["corpus"] = np.array([c] * len(r["utt"]), dtype=object)
        evals.append(r)
        say(f"PREDICT {c}: {len(r['utt'])} rows")
    ev = concat(evals)

    for d in ("models", "heldout", "preds"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        for seed in SEEDS:
            name = f"{cfg['name']}_s{seed}"
            params = {**BASE_PARAMS, "num_leaves": cfg["num_leaves"], "seed": seed}
            x = features(tr, cfg["text"])
            cat = [x.shape[1] - 1] if cfg["text"] else "auto"
            t0 = time.monotonic()
            ds = lgb.Dataset(x, label=tr[LABEL].astype(np.float64), categorical_feature=cat)
            model = lgb.train(params, ds, num_boost_round=cfg["rounds"])
            model.save_model(str(args.out / "models" / f"{name}.txt"))
            ph = model.predict(features(ho, cfg["text"]))
            pq.write_table(
                pa.table(
                    {
                        "source": ho["source"].astype(str),
                        "key": ho["key"].astype(str),
                        "turn": ho["turn"],
                        "hyp_index": ho["hyp_index"],
                        "word_index": ho["word_index"],
                        "label": ho[LABEL].astype(np.int8),
                        "p": ph,
                    }
                ),
                args.out / "heldout" / f"{name}.parquet",
            )
            pe = model.predict(features(ev, cfg["text"]))
            pq.write_table(
                pa.table(
                    {
                        "corpus": ev["corpus"].astype(str),
                        "utt": ev["utt"].astype(str),
                        "turn": ev["turn"],
                        "hyp_index": ev["hyp_index"],
                        "word_index": ev["word_index"],
                        "p": pe,
                    }
                ),
                args.out / "preds" / f"{name}.parquet",
            )
            say(f"FIT {name}: {cfg['rounds']} rounds, {time.monotonic() - t0:.0f} s")
    manifest = {
        "build": build,
        "lightgbm": lgb.__version__,
        "configs": CONFIGS,
        "seeds": list(SEEDS),
        "params": BASE_PARAMS,
        "features": [*NUMERIC, f"text_hash % {TEXT_BUCKETS} (text configurations)"],
        "rows_line": rows_line,
        "held_out": sorted(held),
        "sha256": sha,
        "onpolicy": onpolicy,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
