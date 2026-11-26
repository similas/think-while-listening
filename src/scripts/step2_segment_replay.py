"""Step 2: step 1's selection-set replay again, each hypothesis word's segment recorded.

AMENDMENT 2026-11-26 (segment boundaries), Ali's D2 (a): "GO for the 40-min
segment-index replay; the freeze lifted for it, recorded first; it stops
unless it reproduces step 1's hypotheses word for word." A measured run:
the run skill's launcher, the box quiet.

What it does, per selection-set file (single_step items 20 and up), per turn
as step 1 recorded it (the same origin, open and endpoint, and the same
final, read from the label file and reused, not re-decoded): the live
hypothesis path is replayed as step 1 ran it (twl.label_replay.replay,
LocalAgreement-2 committing, step 1's settings, the runaway guard OFF: step
1's replay predates the guard), with tiny's decode through
twl.stt_engine.transcribe_words_segmented, so each word carries the ordinal
of its decoder segment. Nothing else changes.

THE CHECK, per turn, before the next: the replay must reproduce step 1's
recorded hypotheses word for word: the same number of hypotheses, and for
each the same frozen flag, issue time and committed end, and the same words
(text, start_s, end_s), compared exactly. The first difference stops the run
(STOP line, exit 1); nothing it wrote is used.

THE BOARD, as step2_closed_loop.py's: the replay's clock is simulated
(twl.pacing.modelled_decode_ms), so wall time, clocks and swap enter no
number; tj, swap used, the power mode and the clocks at the start; tj and
swap traffic on every progress line; refuses to start at tj 65 C or above;
holds the run lock; an abort writes its cause before it stops.

VALIDITY, before the first decode: a clean build; the selection set's label
files equal the anchors log's sha256; the replay settings equal step 1's (its
label file's provenance) with the guard off; tiny is the model step 1 used;
faster-whisper and ctranslate2 are the versions step 1's provenance names; the
corpus files' sha256.

Writes --out/segment_replay.jsonl: a header, then one record per file (its
turns, every hypothesis as recorded, each word with its "segment"). The
off-policy count under the segment rule reads it.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_segment_replay.py --anchors results/raw/run_logs/20261124_step2_anchors/anchors.log
        --out results/raw/step2/segment_replay [--corpora ~/corpora] [--limit N] [--log FILE]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import pyarrow.parquet as pq

from scripts.commit_labels import (
    CONFIG,
    REPO,
    RUN_LOCK,
    hold_run_lock,
    model_identity,
    open_corpus,
    replay_settings,
    turn_audio,
    verify_files,
)
from scripts.la_operating_point import provenance
from twl import clocks
from twl.commit import Word
from twl.config import load_config
from twl.label_replay import replay
from twl.provenance import nvpmodel_mode
from twl.step2_segments import replay_difference
from twl.stt_engine import load_pinned, transcribe_words_segmented
from twl.survival_labels import HypWord
from twl.telemetry import read_swaps, read_tj_c, read_vmstat

SEL = "mqa-single_step"
SELECTION_FROM = 20
LABELS = Path("results/raw/commit_labels/num")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--anchors", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corpora", type=Path, default=Path.home() / "corpora")
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="the first N files only: a smoke test")
    args = p.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    build = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if build.endswith("-dirty"):
        raise SystemExit(f"build {build} is dirty: a measured run needs a clean tree")
    out_file = args.out / "segment_replay.jsonl"
    if out_file.exists():
        raise SystemExit(f"{out_file} exists: the replay is not repeated or resumed")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("")

    def say(line: str) -> None:
        print(line, flush=True)
        if args.log:
            with args.log.open("a") as fh:
                fh.write(line + "\n")

    say(f"# step2_segment_replay.py build {build}; {' '.join(sys.argv[1:])}")

    def stop(why: str) -> None:
        say(f"STOP: {why}")
        raise SystemExit(1)

    def board() -> tuple[float, float, int, int]:
        vm = read_vmstat(("pswpin", "pswpout"))
        return (
            read_tj_c(),
            sum(read_swaps().used_mb.values()),
            vm.get("pswpin", 0),
            vm.get("pswpout", 0),
        )

    tj0, swap0, si0, so0 = board()
    say(
        f"BOARD start: tj {tj0:.1f} C; swap used {swap0:.0f} MB; {nvpmodel_mode()}; clocks "
        f"{'pinned' if clocks.pinned() else 'not pinned'}; the replay's clock is simulated, so "
        "wall time, clocks and swap enter no number"
    )
    if tj0 >= 65.0:
        stop(f"tj {tj0:.1f} C at the start (the limit is 65 C)")
    lock_fd = hold_run_lock(REPO / RUN_LOCK)
    # ---- VALIDITY, before the first decode
    anchors = args.anchors.read_text().splitlines()
    rows_path, utts_path = LABELS / f"{SEL}.parquet", LABELS / f"{SEL}.utterances.parquet"
    for f in (rows_path, utts_path):
        if f"# sha256 {sha(f)} {f}" not in anchors:
            stop(f"{f}: sha256 not the anchors log's")
    say("VALIDITY sha256: the selection set's label files equal the anchors log's")
    cfg = load_config(CONFIG)
    settings = replay_settings(cfg)
    prov = provenance(rows_path)
    step1 = dict(prov["settings"]["replay"])  # type: ignore[index, call-overload]
    if settings.runaway_guard or step1.get("runaway_guard", False):
        stop("the guard must be off, as step 1 ran it")
    mine = {k: v for k, v in dataclasses.asdict(settings).items() if k != "runaway_guard"}
    theirs = {k: v for k, v in step1.items() if k != "runaway_guard"}
    if mine != theirs:
        stop(f"replay settings {mine} differ from step 1's {theirs}")
    tiny_name = cfg.stt.partial_model or "tiny"
    if model_identity(tiny_name) != prov["settings"]["models"]["hypothesis"]:  # type: ignore[index]
        stop("tiny is not the model step 1 used")
    say(f"VALIDITY replay: step 1's settings and hypothesis model, the guard off: {mine}")
    pkgs = prov["settings"]["packages"]  # type: ignore[index]
    for name in ("faster-whisper", "ctranslate2"):
        if metadata.version(name) != pkgs[name]:
            stop(f"{name} {metadata.version(name)}, step 1 ran {pkgs[name]}")
    say(
        "VALIDITY decoder: faster-whisper "
        f"{pkgs['faster-whisper']} and ctranslate2 {pkgs['ctranslate2']}, as step 1 ran them"
    )
    corpus = open_corpus(SEL, args.corpora)
    verify_files(args.corpora, corpus.files)
    say("VALIDITY corpus: the files' sha256 verified")
    utts = [
        u
        for u in pq.read_table(utts_path, columns=["utt", "turns"]).to_pylist()
        if int(str(u["utt"]).split(":")[-1]) >= SELECTION_FROM
    ]
    if args.limit:
        utts = utts[: args.limit]
        say(f"SMOKE: the first {len(utts)} files only; nothing reads this run")
    by_uid = {str(u["utt"]): u for u in utts}

    # ---- THE RUN
    cpus = list(cfg.stt.partial_cpus or cfg.stt.cpu_affinity)
    tiny = load_pinned(tiny_name, cfg.stt.partial_cpu_threads, cpus, cfg.stt.compute_type)
    lang = cfg.stt.language

    def decode_hyp(buf, base_s, prompt):  # type: ignore[no-untyped-def]
        raw, wordless = transcribe_words_segmented(
            tiny, buf, language=lang, base_s=base_s, initial_prompt=prompt
        )
        return [
            HypWord(Word(t, s0, s1), pr, alp, nsp, segment=k) for t, s0, s1, pr, alp, nsp, k in raw
        ], wordless

    args.out.mkdir(parents=True, exist_ok=True)
    header = {
        "build": build,
        "args": sys.argv[1:],
        "settings": dataclasses.asdict(settings),
        "files": len(utts),
        "limit": args.limit,
    }
    t_run = time.monotonic()
    tj_max, turns = tj0, 0
    try:
        with out_file.open("w") as fh:
            fh.write(json.dumps({"header": header}) + "\n")
            for done, (uid, pcm) in enumerate(corpus.stream(set(by_uid)), 1):
                u = by_uid[uid]
                recs = []
                for k, t in enumerate(u["turns"]):
                    audio = turn_audio(pcm, float(t["origin_s"]), float(t["endpoint_s"]))
                    final = [
                        Word(str(w["text"]), float(w["start_s"]), float(w["end_s"]))
                        for w in t["final_words"]
                    ]
                    res = replay(
                        audio,
                        final,
                        decode_hyp,
                        settings,
                        float(t["endpoint_s"]) - float(t["origin_s"]),
                        open_s=float(t["opens_s"]) - float(t["origin_s"]),
                    )
                    diff = replay_difference(res.hyps, list(t["hyps"]))
                    if diff is not None:
                        stop(f"{uid} turn {k}: not step 1's hypotheses: {diff}")
                    recs.append({"hyps": res.hyps})
                    turns += 1
                fh.write(json.dumps({"utt": uid, "turns": recs}) + "\n")
                fh.flush()
                if done % 25 == 0 or done == len(utts):
                    tj = read_tj_c()
                    tj_max = max(tj_max, tj)
                    vm = read_vmstat(("pswpin", "pswpout"))
                    say(
                        f"  {done}/{len(utts)} files ({turns} turns) reproduced in "
                        f"{time.monotonic() - t_run:.0f} s; tj {tj:.1f} C (max {tj_max:.1f}); "
                        f"swap traffic pswpin +{vm.get('pswpin', 0) - si0}, pswpout "
                        f"+{vm.get('pswpout', 0) - so0}"
                    )
    except BaseException as e:  # the abort record before the abort
        if not isinstance(e, SystemExit) or e.code not in (0, None, 1):
            say(f"ABORTED: {type(e).__name__}: {e}; files written so far are in {out_file}")
        raise
    tj1, swap1, si1, so1 = board()
    say(
        f"DONE: {len(utts)} files, {turns} turns, every hypothesis step 1's word for word, in "
        f"{time.monotonic() - t_run:.0f} s; tj max {max(tj_max, tj1):.1f} C, now {tj1:.1f} C; "
        f"swap used {swap1:.0f} MB; swap traffic since the start pswpin +{si1 - si0}, pswpout "
        f"+{so1 - so0}; {out_file}"
    )
    del lock_fd


if __name__ == "__main__":
    main()
