"""Step 2's on-policy relabel of LibriSpeech test-clean: the retrain branch's data.

PRE-REGISTRATION 2026-11-24 (345efad), CLOSED LOOP, the retrain branch ("one
on-policy relabel of LibriSpeech test-clean, retrain ..."); Ali's decisions
on 54fce21: "2. (a) GO for the on-policy relabel of LibriSpeech test-clean (≤
3.5 h), driver reviewed and committed first, guarded, τ* 0.64, box quiet.
(b) (i): added to the existing rows."

What it does: the first model (the scored one, as the scoring log names it)
commits at its tau* in the replay (twl.label_replay.replay with
twl.closed_loop.ModelCommitter, the runaway guard on), over every
librispeech-test-clean file step 1 labelled, each turn with step 1's origin,
open, endpoint and final (reused, not re-decoded). The replay's rows
(twl.survival_labels.rows: the features on the model's own stream, label_num
against the turn's final) are the on-policy label rows.

THE BOARD and the abort record as in step2_closed_loop.py: tj, swap used,
the power mode and the clocks at the start; tj and the swap traffic on
every progress line; the totals at the end; no start at tj 65 C or above;
the log written line by line; SIGTERM and every exception write an ABORTED
line first.

VALIDITY, before the first decode: a clean build; the model, seed and tau
equal those the scoring log scored; the replay settings and the hypothesis
model equal step 1's for librispeech-test-clean (its label file's
provenance), with the guard on; the model reproduces its scored P on every
selection-set row through this script's feature path; step 1's label files
for the corpus (the turns, finals and utt ids reused) equal the sha256 the
first model's training manifest records; the corpus files' sha256.

Writes, under --out:
- librispeech-test-clean.onpolicy.jsonl: a header, then one record per file
  (its turns' hypotheses, each with the fresh indices the model committed,
  and its rows), written as it goes;
- librispeech-test-clean.onpolicy.parquet: every row, in step 1's row
  schema (scripts.commit_labels.ROW_SCHEMA), written file by file to a
  .partial name and renamed only once the run completes;
- librispeech-test-clean.onpolicy.utterances.parquet: one row per file,
  its turns (step 1's origin, open, endpoint and final words) with their
  hypotheses (frozen, words, runaway, committed_idx), in a fixed schema
  (UTT_SCHEMA), written file by file to a .partial name and renamed with
  the rows, so that step2_train.guarded_out drops the guard's hypotheses
  from these rows as from step 1's.
Both parquet files carry twl_provenance as step 1's do (the build, the run
metadata, the settings with the replay's, the committing model, the source
files' sha256), readable by la_operating_point.provenance. The utt ids are
step 1's, so the training can key these rows as
f"librispeech-test-clean:{utt}" and the held-out split does not change.

THE CAP (Ali's GO: "≤ 3.5 h"): every progress line states the time against
--cap-s (12600 s); past it, the run stops with an ABORTED record and no
parquet, to be reported, not continued without Ali.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_relabel.py --models results/raw/step2/primary --config L63_R200_text --seed 2
        --tau 0.64 --score results/raw/run_logs/20261124_step2_score/step2_score.log
        --out results/raw/commit_labels/onpolicy [--corpora ~/corpora] [--log FILE] [--limit N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shlex
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.commit_labels import (
    CONFIG,
    REPO,
    ROW_SCHEMA,
    RUN_LOCK,
    STACK,
    _with_meta,
    hold_run_lock,
    model_identity,
    open_corpus,
    replay_settings,
    turn_audio,
    verify_files,
)
from scripts.la_operating_point import provenance
from scripts.step2_closed_loop import predictor, sha
from twl import clocks
from twl.closed_loop import ModelCommitter
from twl.commit import Word
from twl.config import load_config
from twl.label_replay import replay
from twl.provenance import build_run_meta, new_run_id, nvpmodel_mode
from twl.stt_engine import load_pinned, transcribe_words_scored
from twl.survival_labels import LABEL_VERSION, HypWord
from twl.telemetry import read_swaps, read_tj_c, read_vmstat

CORPUS = "librispeech-test-clean"
SEL = "mqa-single_step"
SELECTION_FROM = 20
LABELS = Path("results/raw/commit_labels/num")
MODULES = (
    "label_replay", "survival_labels", "stt_engine", "pacing", "commit", "seal", "label_io",
    "wer", "closed_loop", "runaway",
)  # fmt: skip
_WORDS = pa.list_(
    pa.struct([("text", pa.string()), ("start_s", pa.float64()), ("end_s", pa.float64())])
)
_HYP = pa.struct(
    [
        ("issued_s", pa.float64()),
        ("committed_end_s", pa.float64()),
        ("buffer_s", pa.float64()),
        ("modelled_ms", pa.float64()),
        ("wall_ms", pa.float64()),
        ("frozen", pa.bool_()),
        ("words", _WORDS),
        ("runaway", pa.string()),
        ("committed_idx", pa.list_(pa.int64())),
    ]
)
UTT_SCHEMA = pa.schema(
    [
        ("utt", pa.string()),
        (
            "turns",
            pa.list_(
                pa.struct(
                    [
                        ("origin_s", pa.float64()),
                        ("opens_s", pa.float64()),
                        ("endpoint_s", pa.float64()),
                        ("final_words", _WORDS),
                        ("hyps", pa.list_(_HYP)),
                        ("rows", pa.int64()),
                    ]
                )
            ),
        ),
    ]
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--score", type=Path, required=True, help="step2_score.py's log")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corpora", type=Path, default=Path.home() / "corpora")
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="the first N files only: a smoke test")
    p.add_argument("--cap-s", type=float, default=12600.0, help="Ali's cap: 3.5 h")
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
    stem = args.out / f"{CORPUS}.onpolicy"
    out_jsonl = args.out / f"{stem.name}.jsonl"
    out_parquet = args.out / f"{stem.name}.parquet"
    out_partial = args.out / f"{stem.name}.parquet.partial"
    out_utts = args.out / f"{stem.name}.utterances.parquet"
    out_utts_partial = args.out / f"{stem.name}.utterances.parquet.partial"
    if any(f.exists() for f in (out_jsonl, out_parquet, out_partial, out_utts, out_utts_partial)):
        raise SystemExit(f"{stem}.*: exists; a relabel is not repeated or resumed")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("")

    def say(line: str) -> None:
        print(line, flush=True)
        if args.log:
            with args.log.open("a") as fh:
                fh.write(line + "\n")

    def stop(why: str) -> None:
        say(f"STOP: {why}")
        raise SystemExit(1)

    def swap_io() -> tuple[int, int]:
        vm = read_vmstat(("pswpin", "pswpout"))
        return vm.get("pswpin", 0), vm.get("pswpout", 0)

    say(f"# step2_relabel.py build {build}; {' '.join(sys.argv[1:])}")
    tj0 = read_tj_c()
    si0, so0 = swap_io()
    say(
        f"BOARD start: tj {tj0:.1f} C; swap used {sum(read_swaps().used_mb.values()):.0f} MB; "
        f"{nvpmodel_mode()}; clocks {'pinned' if clocks.pinned() else 'not pinned'}; the "
        "replay's clock is simulated, so wall time, clocks and swap enter no number"
    )
    if tj0 >= 65.0:
        stop(f"tj {tj0:.1f} C at the start (the limit is 65 C)")
    lock_fd = hold_run_lock(REPO / RUN_LOCK)  # the fence, both ways

    # ---- VALIDITY, before the first decode
    score = args.score.read_text()
    chosen = re.search(r"-> (\S+) \(lowest mean held-out log-loss", score)
    mseed = re.search(r"^MEDIAN SEED (\d+);", score, re.M)
    mtau = re.search(r"^PRIMARY guarded \(scored\): tau\* ([0-9.]+);", score, re.M)
    scored_as = (
        (chosen.group(1), int(mseed.group(1)), float(mtau.group(1)))
        if chosen and mseed and mtau
        else None
    )
    asked = (args.config, args.seed, args.tau)
    if scored_as != asked:
        stop(f"the scoring log scored {scored_as}; this run asks for {asked}")
    say(f"VALIDITY model choice: {scored_as}, as the scoring log scored")
    cfg = load_config(CONFIG)
    settings = dataclasses.replace(replay_settings(cfg), runaway_guard=True)
    rows_path, utts_path = LABELS / f"{CORPUS}.parquet", LABELS / f"{CORPUS}.utterances.parquet"
    prov = provenance(rows_path)
    mine = {k: v for k, v in dataclasses.asdict(settings).items() if k != "runaway_guard"}
    theirs = {k: v for k, v in prov["settings"]["replay"].items() if k != "runaway_guard"}  # type: ignore[index,union-attr]
    if mine != theirs:
        stop(f"replay settings {mine} differ from step 1's {theirs}")
    tiny_name = cfg.stt.partial_model or "tiny"
    if model_identity(tiny_name) != prov["settings"]["models"]["hypothesis"]:  # type: ignore[index]
        stop("tiny is not the model step 1 used")
    say(f"VALIDITY replay: step 1's {CORPUS} settings and hypothesis model, guard on: {mine}")
    manifest = json.loads((args.models / "manifest.json").read_text())
    source_sha = {str(f): sha(f) for f in (rows_path, utts_path)}
    for f, h in source_sha.items():
        if manifest["sha256"].get(f) != h:
            stop(f"{f}: sha256 {h}, not the {manifest['sha256'].get(f)} the model trained on")
    say(
        f"VALIDITY step 1's label files: the sha256 the training manifest records ({rows_path}: "
        f"{source_sha[str(rows_path)][:12]}..., {utts_path}: {source_sha[str(utts_path)][:12]}...)"
    )
    cfg_of = {c["name"]: c for c in manifest["configs"]}
    model_file = args.models / "models" / f"{args.config}_s{args.seed}.txt"
    booster = lgb.Booster(model_file=str(model_file))
    predict = predictor(booster, bool(cfg_of[args.config]["text"]))
    sel_rows = [
        r
        for r in pq.read_table(LABELS / f"{SEL}.parquet").to_pylist()
        if int(str(r["utt"]).split(":")[-1]) >= SELECTION_FROM
    ]
    scored = {
        (str(r["utt"]), int(r["turn"]), int(r["hyp_index"]), int(r["word_index"])): float(r["p"])
        for r in pq.read_table(
            args.models / "preds" / f"{args.config}_s{args.seed}.parquet"
        ).to_pylist()
        if r["corpus"] == SEL
    }
    worst = max(
        abs(q - scored[(str(r["utt"]), int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))])
        for r, q in zip(sel_rows, predict(sel_rows), strict=True)
    )
    if worst > 1e-9:
        stop(f"the model's P through this feature path differs from its scored P by {worst}")
    say(
        f"VALIDITY model: {model_file} (sha256 {sha(model_file)}) reproduces its scored P on "
        f"{len(sel_rows)} selection-set rows (largest difference {worst:.1e})"
    )
    corpus = open_corpus(CORPUS, args.corpora)
    verify_files(args.corpora, corpus.files)
    say("VALIDITY corpus: the files' sha256 verified")
    utts = pq.read_table(utts_path, columns=["utt", "turns"]).to_pylist()
    if args.limit:
        utts = utts[: args.limit]
        say(f"SMOKE: the first {len(utts)} files only")
    by_uid = {str(u["utt"]): u for u in utts}

    # ---- THE RUN
    cpus = list(cfg.stt.partial_cpus or cfg.stt.cpu_affinity)
    tiny = load_pinned(tiny_name, cfg.stt.partial_cpu_threads, cpus, cfg.stt.compute_type)
    lang = cfg.stt.language

    def decode_hyp(buf, base_s, prompt):  # type: ignore[no-untyped-def]
        raw, wordless = transcribe_words_scored(
            tiny, buf, language=lang, base_s=base_s, initial_prompt=prompt
        )
        return [HypWord(Word(t, s0, s1), pr, alp, nsp) for t, s0, s1, pr, alp, nsp in raw], wordless

    args.out.mkdir(parents=True, exist_ok=True)
    header = {
        "build": build,
        "args": sys.argv[1:],
        "corpus": CORPUS,
        "settings": dataclasses.asdict(settings),
        "model_file": str(model_file),
        "model_sha256": sha(model_file),
        "config": args.config,
        "seed": args.seed,
        "tau": args.tau,
        "files": len(utts),
        "limit": args.limit,
        "cap_s": args.cap_s,
        "label_files_sha256": source_sha,
    }
    meta = build_run_meta(
        run_id=new_run_id("onpolicy"),
        config_path=CONFIG,
        notes="step2_relabel " + CORPUS,
        extra_software={p: metadata.version(p) for p in (*STACK, "lightgbm")},
    )
    step1_settings = prov["settings"]
    assert isinstance(step1_settings, dict)
    record = {
        **step1_settings,  # the turns, labels and features as step 1 records them
        "script_sha256": sha(REPO / "src/scripts/step2_relabel.py"),
        "module_sha256": {m: sha(REPO / f"src/twl/{m}.py") for m in MODULES},
        "packages": {p: metadata.version(p) for p in (*STACK, "lightgbm")},
        "replay": dataclasses.asdict(settings),
        "label_num_version": LABEL_VERSION,
    }
    run_prov: dict[str, object] = {
        "invocations": [{"run_meta": dataclasses.asdict(meta), "command": shlex.join(sys.argv)}],
        "settings": record,
        "corpus": CORPUS,
        "role": prov.get("role"),
        "onpolicy": {k: v for k, v in header.items() if k != "settings"},
        "step1_provenance": {
            k: prov.get(k) for k in ("invocations", "relabel", "corpus_files_sha256")
        },
    }
    t_run = time.monotonic()
    tj_max, done, n_rows = tj0, 0, 0
    try:
        with (
            out_jsonl.open("w") as fh,
            pq.ParquetWriter(out_partial, _with_meta(ROW_SCHEMA, run_prov)) as writer,
            pq.ParquetWriter(out_utts_partial, _with_meta(UTT_SCHEMA, run_prov)) as utt_writer,
        ):
            fh.write(json.dumps({"header": header}) + "\n")
            for uid, pcm in corpus.stream(set(by_uid)):
                if time.monotonic() - t_run > args.cap_s:
                    stop(f"past the cap of {args.cap_s:.0f} s after {done} files")
                u = by_uid[uid]
                turn_recs, rows = [], []
                for k, t in enumerate(u["turns"]):
                    origin, endpoint = float(t["origin_s"]), float(t["endpoint_s"])
                    final = [
                        Word(str(w["text"]), float(w["start_s"]), float(w["end_s"]))
                        for w in t["final_words"]
                    ]
                    m = ModelCommitter(
                        predict=predict, tau=args.tau, tail_guard_s=settings.tail_guard_s
                    )
                    res = replay(
                        turn_audio(pcm, origin, endpoint),
                        final,
                        decode_hyp,
                        settings,
                        endpoint - origin,
                        open_s=float(t["opens_s"]) - origin,
                        model=m,
                    )
                    rows += [dict(r, utt=uid, turn=k, turn_origin_s=origin) for r in res.rows]
                    turn_recs.append(
                        {
                            "origin_s": origin,
                            "opens_s": float(t["opens_s"]),
                            "endpoint_s": endpoint,
                            "final_words": t["final_words"],
                            "hyps": res.hyps,
                            "rows": len(res.rows),
                        }
                    )
                fh.write(json.dumps({"utt": uid, "turns": turn_recs, "rows": rows}) + "\n")
                fh.flush()
                writer.write_table(pa.Table.from_pylist(rows, schema=writer.schema))
                utt_writer.write_table(
                    pa.Table.from_pylist(
                        [{"utt": uid, "turns": turn_recs}], schema=utt_writer.schema
                    )
                )
                n_rows += len(rows)
                done += 1
                if done % 100 == 0 or done == len(utts):
                    tj = read_tj_c()
                    tj_max = max(tj_max, tj)
                    si, so = swap_io()
                    say(
                        f"  {done}/{len(utts)} files in {time.monotonic() - t_run:.0f} s "
                        f"(cap {args.cap_s:.0f} s); {n_rows} rows; tj {tj:.1f} C (max "
                        f"{tj_max:.1f}); swap traffic pswpin +{si - si0}, pswpout +{so - so0}"
                    )
        if done != len(utts):
            stop(f"{done} files relabelled of {len(utts)}")
        out_partial.rename(out_parquet)
        out_utts_partial.rename(out_utts)
    except BaseException as e:  # the abort record before the abort
        say(f"ABORTED: {type(e).__name__}: {e}; {done} files written to {out_jsonl}")
        raise
    si, so = swap_io()
    say(
        f"DONE: {done} files, {n_rows} rows in {time.monotonic() - t_run:.0f} s; tj max "
        f"{max(tj_max, read_tj_c()):.1f} C; swap traffic since the start pswpin +{si - si0}, "
        f"pswpout +{so - so0}"
    )
    for f in (out_parquet, out_utts):
        say(f"# sha256 {sha(f)} {f}")
    del lock_fd


if __name__ == "__main__":
    main()
