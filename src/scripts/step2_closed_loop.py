"""Step 2's closed loop: one guarded replay of the selection set, the model committing.

PRE-REGISTRATION 2026-11-24 (345efad), CLOSED LOOP (decision 8); Ali's GO
(2026-11-24): "GO for the closed-loop run on the scored model
(L63_R200_text, seed 2, tau* 0.64), guarded, ~1 h, box quiet, as soon as the
review clears". A measured run: the run skill's launcher, the box quiet.

What it does, per selection-set file (single_step items 20 and up), per turn
as step 1 recorded it (the same origin, open and endpoint, and the same
final, read from the label file: neither depends on the committer, so they
are reused, not re-decoded): the live hypothesis path is replayed
(twl.label_replay.replay, the deployed settings, the runaway guard on), with
the learned committer in LA-2's place (twl.closed_loop.ModelCommitter: the
longest prefix with P >= tau, features from this stream, LA-2 in shadow).
tiny decodes every hypothesis again, since the buffers now start at the
model's committed end.

THE BOARD: the replay's clock is simulated (twl.pacing.modelled_decode_ms),
so wall time, clocks and swap enter no number; they are logged so the run's
conditions are on record (tj, swap used, the power mode and the clocks at the
start; tj and swap traffic, pswpin/pswpout since the start, on every progress
line; the maximum tj and the total swap traffic at the end). It refuses to
start at tj 65 C or above. The log is written line by line, and an abort
writes its cause before it stops.

VALIDITY, before the first decode: a clean build; the configuration, seed
and tau equal those the scoring log scored; the selection set's label files
equal the anchors log's sha256; the replay settings equal step 1's
(its label file's provenance), with the guard on; tiny is the model step 1
used; the corpus files' sha256; and the scored model, loaded from its file,
reproduces its scored predictions on every selection-set row through this
script's feature path (the training script's features()).

Writes --out/closed_loop.jsonl: a header, then one record per file (its
turns, every hypothesis with the fresh indices the model committed, and the
final's words). step2_closed_loop_score.py scores it.

WITH --reproduce RUN (Ali's 2 (a) of 2026-11-27): a re-run of an earlier
loop of the plain rule (the retrained model's, a6ac6d5) to record what that
loop did not: each fresh word's P and where the prefix rule stopped, and
each word's segment (the segmented decode, whose words are the scored
decode's, tested). Before the first decode, the earlier run's header must
match: the same model file (sha256), tau, replay settings, no segment rule,
not a smoke run, a clean build. After each turn, the turn must reproduce the
earlier run's word for word, times included: the number of hypotheses and,
per hypothesis, its frozen flag, issue time and committed end, its words'
text, start and end, and the fresh indices the model committed
(twl.step2_segments.replay_difference, committed=True). The first
difference stops the run (a STOP line, exit 1). The new records are not
compared, nor wall_ms and the runaway verdict. It adds no verdict.

WITH --segments (the AMENDMENT of 2026-11-26): tiny decodes through
twl.stt_engine.transcribe_words_segmented, so each word carries its segment;
the committer cuts its prefix back to the last segment boundary (item 2);
every offered hypothesis records each fresh word's P, where the prefix rule
stopped and where the cut fell (item 8); the header says segments true. Its
--score is step2_segment_score.py's log (tau* under the rule), and the
scoring log must be that script's, and only with --segments.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_closed_loop.py --models results/raw/step2/primary --config L63_R200_text --seed 2
        --tau 0.64 --anchors results/raw/run_logs/20261124_step2_anchors/anchors.log
        --out results/raw/step2/closed_loop [--corpora ~/corpora] [--log FILE]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import lightgbm as lgb
import numpy as np
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
from scripts.step2_train import NUMERIC, features
from twl import clocks
from twl.closed_loop import ModelCommitter, Row
from twl.commit import Word
from twl.config import load_config
from twl.label_replay import replay
from twl.provenance import nvpmodel_mode
from twl.step2_segments import replay_difference
from twl.stt_engine import load_pinned, transcribe_words_scored, transcribe_words_segmented
from twl.survival_labels import HypWord
from twl.telemetry import read_swaps, read_tj_c, read_vmstat

SEL = "mqa-single_step"
SELECTION_FROM = 20
LABELS = Path("results/raw/commit_labels/num")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictor(booster: lgb.Booster, text: bool):  # type: ignore[no-untyped-def]
    """P(survive) for rows (dicts), through the training script's features()."""

    def predict(rows: Sequence[Row]) -> list[float]:
        if not rows:
            return []
        cols = {c: np.array([r[c] for r in rows]) for c in (*NUMERIC, "text_hash")}
        return [float(x) for x in booster.predict(features(cols, text))]

    return predict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--anchors", type=Path, required=True)
    p.add_argument("--score", type=Path, required=True, help="step2_score.py's log")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corpora", type=Path, default=Path.home() / "corpora")
    p.add_argument("--log", type=Path, default=None)
    p.add_argument(
        "--limit", type=int, default=0, help="the first N files only: a smoke test, never scored"
    )
    p.add_argument(
        "--segments", action="store_true", help="the segment rule (AMENDMENT 2026-11-26, item 2)"
    )
    p.add_argument(
        "--reproduce",
        type=Path,
        default=None,
        help="an earlier plain-rule loop's closed_loop.jsonl, to reproduce word for word",
    )
    args = p.parse_args()
    # SIGTERM (the launcher's stop, the watchdog's) raises SystemExit, so the
    # abort record below is written before the process ends.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    build = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if build.endswith("-dirty"):
        raise SystemExit(f"build {build} is dirty: a measured run needs a clean tree")
    out_file = args.out / "closed_loop.jsonl"
    if out_file.exists():
        raise SystemExit(f"{out_file} exists: a closed-loop run is not repeated or resumed")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("")

    def say(line: str) -> None:
        print(line, flush=True)
        if args.log:
            with args.log.open("a") as fh:
                fh.write(line + "\n")

    say(f"# step2_closed_loop.py build {build}; {' '.join(sys.argv[1:])}")

    def stop(why: str) -> None:
        say(f"STOP: {why}")
        raise SystemExit(1)

    def board() -> tuple[float, float, int, int]:
        """tj (C), swap used (MB), pswpin, pswpout."""
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
    lock_fd = hold_run_lock(REPO / RUN_LOCK)  # the fence, both ways
    # ---- VALIDITY, before the first decode
    anchors = args.anchors.read_text().splitlines()
    rows_path, utts_path = LABELS / f"{SEL}.parquet", LABELS / f"{SEL}.utterances.parquet"
    for f in (rows_path, utts_path):
        if f"# sha256 {sha(f)} {f}" not in anchors:
            stop(f"{f}: sha256 not the anchors log's")
    say("VALIDITY sha256: the selection set's label files equal the anchors log's")
    score = args.score.read_text()
    by_segment_scorer = score.startswith("# step2_segment_score.py")
    if by_segment_scorer != args.segments:
        stop(f"--segments is {args.segments}, but the scoring log is {score.split(';')[0]!r}")
    chosen = re.search(r"-> (\S+) \(lowest mean held-out log-loss", score)
    mseed = re.search(r"^MEDIAN SEED (\d+);", score, re.M)
    mtau = re.search(r"^PRIMARY guarded \(scored\): tau\* ([0-9.]+);", score, re.M)
    scored_as = (
        (chosen.group(1), int(mseed.group(1)), float(mtau.group(1)))
        if chosen and mseed and mtau
        else None
    )
    if scored_as != (args.config, args.seed, args.tau):
        asked = (args.config, args.seed, args.tau)
        stop(f"the scoring log scored {scored_as}; this run asks for {asked}")
    say(f"VALIDITY model choice: {scored_as}, as the scoring log scored")
    cfg = load_config(CONFIG)
    settings = dataclasses.replace(replay_settings(cfg), runaway_guard=True)
    prov = provenance(rows_path)
    step1 = prov["settings"]["replay"]  # type: ignore[index]
    mine = {k: v for k, v in dataclasses.asdict(settings).items() if k != "runaway_guard"}
    theirs = {k: v for k, v in step1.items() if k != "runaway_guard"}  # type: ignore[union-attr]
    if mine != theirs:
        stop(f"replay settings {mine} differ from step 1's {theirs}")
    tiny_name = cfg.stt.partial_model or "tiny"
    if model_identity(tiny_name) != prov["settings"]["models"]["hypothesis"]:  # type: ignore[index]
        stop("tiny is not the model step 1 used")
    say(f"VALIDITY replay: step 1's settings and hypothesis model, with the guard on: {mine}")
    manifest = json.loads((args.models / "manifest.json").read_text())
    cfg_of = {c["name"]: c for c in manifest["configs"]}
    if args.config not in cfg_of or args.seed not in manifest["seeds"]:
        stop(f"{args.config} seed {args.seed} is not in the manifest")
    model_file = args.models / "models" / f"{args.config}_s{args.seed}.txt"
    booster = lgb.Booster(model_file=str(model_file))
    predict = predictor(booster, bool(cfg_of[args.config]["text"]))
    rows = pq.read_table(rows_path).to_pylist()
    rows = [r for r in rows if int(str(r["utt"]).split(":")[-1]) >= SELECTION_FROM]
    scored = {
        (str(r["utt"]), int(r["turn"]), int(r["hyp_index"]), int(r["word_index"])): float(r["p"])
        for r in pq.read_table(
            args.models / "preds" / f"{args.config}_s{args.seed}.parquet"
        ).to_pylist()
        if r["corpus"] == SEL
    }
    mine_p = predict(rows)
    worst = max(
        abs(q - scored[(str(r["utt"]), int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))])
        for r, q in zip(rows, mine_p, strict=True)
    )
    if worst > 1e-9:
        stop(f"the model's P through this feature path differs from its scored P by {worst}")
    say(
        f"VALIDITY model: {model_file} (sha256 {sha(model_file)}) reproduces its scored P on "
        f"{len(rows)} rows (largest difference {worst:.1e})"
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
        say(f"SMOKE: the first {len(utts)} files only; the scorer refuses this run")
    by_uid = {str(u["utt"]): u for u in utts}

    # ---- THE RUN
    cpus = list(cfg.stt.partial_cpus or cfg.stt.cpu_affinity)
    tiny = load_pinned(tiny_name, cfg.stt.partial_cpu_threads, cpus, cfg.stt.compute_type)
    lang = cfg.stt.language

    def decode_hyp(buf, base_s, prompt):  # type: ignore[no-untyped-def]
        if args.segments or args.reproduce:
            seg, wordless = transcribe_words_segmented(
                tiny, buf, language=lang, base_s=base_s, initial_prompt=prompt
            )
            return [
                HypWord(Word(t, s0, s1), pr, alp, nsp, segment=k)
                for t, s0, s1, pr, alp, nsp, k in seg
            ], wordless
        raw, wordless = transcribe_words_scored(
            tiny, buf, language=lang, base_s=base_s, initial_prompt=prompt
        )
        return [HypWord(Word(t, s0, s1), pr, alp, nsp) for t, s0, s1, pr, alp, nsp in raw], wordless

    earlier: dict[str, list[dict[str, object]]] | None = None
    if args.reproduce:
        if args.segments:
            stop("--reproduce re-runs a plain-rule loop: not with --segments")
        old = [json.loads(x) for x in args.reproduce.read_text().splitlines() if x.strip()]
        oh = old[0]["header"]
        want = {
            "model_sha256": sha(model_file),
            "tau": args.tau,
            "settings": dataclasses.asdict(settings),
            "segments": False,
            "limit": 0,
        }
        got = {k: oh.get(k, False if k == "segments" else None) for k in want}
        if got != want or str(oh["build"]).endswith("-dirty"):
            stop(f"the earlier run's header {got} (build {oh['build']}) is not this run's {want}")
        earlier = {str(r["utt"]): r["turns"] for r in old[1:]}
        if not set(by_uid) <= set(earlier) or (not args.limit and set(by_uid) != set(earlier)):
            stop("the earlier run's files are not this run's")
        say(
            f"VALIDITY reproduce: {args.reproduce} (build {str(oh['build'])[:7]}, clean): the same "
            f"model file, tau {args.tau}, replay settings, no segment rule; {len(earlier)} files; "
            "each turn must reproduce it word for word, times and commits included"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    header = {
        "build": build,
        "args": sys.argv[1:],
        "settings": dataclasses.asdict(settings),
        "model_file": str(model_file),
        "model_sha256": sha(model_file),
        "tau": args.tau,
        "files": len(utts),
        "limit": args.limit,
        "segments": args.segments,
        "reproduce": str(args.reproduce) if args.reproduce else None,
    }
    t_run = time.monotonic()
    try:
        tj_max = run_files(
            out_file, header, corpus, by_uid, utts, decode_hyp, predict, settings, args, earlier
        )
    except BaseException as e:  # the abort record before the abort
        if not (isinstance(e, SystemExit) and e.code == 1):  # a STOP line is its own record
            say(f"ABORTED: {type(e).__name__}: {e}; files written so far are in {out_file}")
        raise
    tj1, swap1, si1, so1 = board()
    say(
        f"DONE: {len(utts)} files in {time.monotonic() - t_run:.0f} s; tj max "
        f"{max(tj_max, tj1):.1f} C, now {tj1:.1f} C; swap used {swap1:.0f} MB; swap traffic "
        f"since the start pswpin "
        f"+{si1 - si0}, pswpout +{so1 - so0}; {out_file}"
    )
    del lock_fd


def run_files(  # type: ignore[no-untyped-def]
    out_file, header, corpus, by_uid, utts, decode_hyp, predict, settings, args, earlier=None
) -> float:
    """The replay over the files, checkpointing each; progress with the board.
    Returns the maximum tj read."""
    t_run = time.monotonic()
    tj0, si0, so0 = (
        read_tj_c(),
        *(read_vmstat(("pswpin", "pswpout")).get(k, 0) for k in ("pswpin", "pswpout")),
    )
    tj_max = tj0
    done = 0
    with out_file.open("w") as fh:
        fh.write(json.dumps({"header": header}) + "\n")
        for uid, pcm in corpus.stream(set(by_uid)):
            u = by_uid[uid]
            turn_recs = []
            for k, t in enumerate(u["turns"]):
                audio = turn_audio(pcm, float(t["origin_s"]), float(t["endpoint_s"]))
                final = [
                    Word(str(w["text"]), float(w["start_s"]), float(w["end_s"]))
                    for w in t["final_words"]
                ]
                m = ModelCommitter(
                    predict=predict,
                    tau=args.tau,
                    tail_guard_s=settings.tail_guard_s,
                    segments=args.segments,
                )
                res = replay(
                    audio,
                    final,
                    decode_hyp,
                    settings,
                    float(t["endpoint_s"]) - float(t["origin_s"]),
                    open_s=float(t["opens_s"]) - float(t["origin_s"]),
                    model=m,
                )
                if earlier is not None:
                    old_turns = earlier[uid]
                    diff = (
                        f"{len(u['turns'])} turns, earlier {len(old_turns)}"
                        if len(old_turns) != len(u["turns"])
                        else replay_difference(res.hyps, old_turns[k]["hyps"], committed=True)
                    )
                    if diff is not None:
                        line = f"STOP: {uid} turn {k}: not the earlier run's: {diff}"
                        print(line, flush=True)
                        if args.log:
                            with args.log.open("a") as lf:
                                lf.write(line + "\n")
                        raise SystemExit(1)
                turn_recs.append(
                    {
                        "origin_s": t["origin_s"],
                        "opens_s": t["opens_s"],
                        "endpoint_s": t["endpoint_s"],
                        "final_words": t["final_words"],
                        "hyps": res.hyps,
                        "rows": len(res.rows),
                    }
                )
            fh.write(json.dumps({"utt": uid, "turns": turn_recs}) + "\n")
            fh.flush()
            done += 1
            if done % 25 == 0 or done == len(utts):
                tj = read_tj_c()
                tj_max = max(tj_max, tj)
                vm = read_vmstat(("pswpin", "pswpout"))
                line = (
                    f"  {done}/{len(utts)} files in {time.monotonic() - t_run:.0f} s; tj "
                    f"{tj:.1f} C (max {tj_max:.1f}); swap traffic pswpin "
                    f"+{vm.get('pswpin', 0) - si0}, pswpout +{vm.get('pswpout', 0) - so0}"
                )
                print(line, flush=True)
                if args.log:
                    with args.log.open("a") as lf:
                        lf.write(line + "\n")
    return tj_max


if __name__ == "__main__":
    main()
