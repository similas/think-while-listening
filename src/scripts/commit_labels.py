"""Survival labels for the learned committer (phase step 1): the I/O around twl.label_replay.

For every utterance of a corpus: the FINAL is base on the full utterance with
word timestamps (twl.stt_engine.transcribe_words_scored, the live decode call);
tiny's hypotheses are replayed as the live issue path would issue them, on a
simulated clock (twl.label_replay; its docstring states every departure); and
each fresh word is labelled and given its features (twl.survival_labels).

Run in ~/.venvs/twl-learn (the measurement venv's recognizer stack, plus
pyarrow) with PYTHONPATH=src, from the repository root:

    commit_labels.py --corpus librispeech-test-clean --limit 20 --tag pilot
    commit_labels.py --corpus peoples-speech --approved "<Ali's GO, with its date>"

A whole corpus (no --limit) needs --approved: CLAUDE.md §6 asks before any run
over 30 minutes, and a pilot gives the projection first.

Outputs in results/raw/commit_labels/ (or --out): <stem>.parquet, one row per
labelled word; <stem>.utterances.parquet, one row per utterance with the
final's timed words; <stem>.jsonl, the checkpoint. Its first line records the
settings; a resume with other settings stops. Provenance is in each parquet
file's key-value metadata under "twl_provenance".

THE SEAL. multi_step's 80 sealed rows are filtered out by uid before any audio
is decoded (twl.seal), after the seal's hash and the parquet's row order are
checked; either check failing stops the run. Their bytes pass through the
parquet reader undecoded.

The final LABELS and COST lines are this script's output, read by people and
by the claim registry, and are printed; progress goes to logging.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import logging
import os
import random
import re
import resource
import shlex
import statistics
import sys
import tarfile
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from twl.commit import Word
from twl.config import TwlConfig, load_config
from twl.label_io import (
    PILOT_MAX,
    LabelRunError,
    apply_seal,
    check_sha256,
    hold_run_lock,
    parse_sha256sums,
    read_checkpoint,
    refuse_dirty,
    require_approval,
)
from twl.label_replay import ReplaySettings, replay
from twl.pacing import PARTIAL_INTERCEPT_MS, PARTIAL_MS_PER_S
from twl.provenance import build_run_meta, new_run_id
from twl.seal import SEAL_SHA256
from twl.stt_engine import MIN_DECODE_S, POLL_S, load_pinned, transcribe_words_scored
from twl.survival_labels import GAP_COST, MATCH_TOL_S, SUB_COST, WINDOW_S, HypWord
from twl.wer import normalize, wer

log = logging.getLogger("commit_labels")
REPO = Path(__file__).resolve().parents[2]
CONFIG = Path("src/configs/commit_wl_stop_gate.yaml")
SEAL = Path("results/raw/spoken_mqa/multi_step_fresh80-seed20261118.json")
MQA_INDEX = Path("results/raw/spoken_mqa/multi_step_reasoning_index.json")
FETCH = Path("src/scripts/fetch_corpora.sh")
OUT = Path("results/raw/commit_labels")
RAW = Path("results/raw")
RUN_LOCK = RAW / ".run.lock"
LABEL_COLUMNS = ("label", "label_wer_norm", "match_offset_s")  # read the final
ID_COLUMNS = ("utt", "hyp_index", "word_index", "text", "key", "issued_s", "start_s", "end_s")
STACK = ("faster-whisper", "ctranslate2", "numpy", "onnxruntime", "pyarrow", "av")
SR = 16000
PROMPT_CHARS = 200  # LocalAgreementCommitter.prompt's default, as live
BOOT = 2000

MQA_SPLITS = {
    "mqa-short_digit": "short_digit",
    "mqa-long_digit": "long_digit",
    "mqa-single_step": "single_step_reasoning",
    "mqa-multi_step": "multi_step_reasoning",
}
CORPORA = ("librispeech-test-clean", "librispeech-test-other", "peoples-speech", *MQA_SPLITS)
# Ali's step 2 and the labels skill's table. Training refuses any other role.
ROLE = {
    "librispeech-test-clean": "train",
    "librispeech-test-other": "train",
    "peoples-speech": "train",
    "mqa-single_step": "validation and selection (OPEN in NOTES: the 2026-10-01 sentence)",
    "mqa-short_digit": "offline evaluation",
    "mqa-long_digit": "offline evaluation",
    "mqa-multi_step": "offline evaluation only; never train or validate",
}
Pcm = npt.NDArray[np.float32]


# ------------------------------------------------------------------ corpora


@dataclass(frozen=True)
class Corpus:
    """A corpus's utterance list (cheap to build) and a stream of its audio."""

    refs: list[tuple[str, str]]  # (uid, reference), in file order
    files: list[Path]  # the files the audio comes from (as SHA256SUMS names them)
    stream: Callable[[set[str]], Iterator[tuple[str, Pcm]]]  # wanted uids, file order


def _pcm(data: bytes | str) -> Pcm:
    """16 kHz float32 PCM from encoded audio bytes or a file path."""
    src = io.BytesIO(data) if isinstance(data, bytes) else data
    pcm: Pcm = decode_audio(src, sampling_rate=SR)
    return pcm


def librispeech(corpora: Path, split: str) -> Corpus:
    """LibriSpeech ``split`` from its extracted tree, checked against its archive.

    Raises:
        SystemExit: the tree is missing, or holds a different number of
            utterances than the archive whose sha256 is pinned.
    """
    archive = corpora / "librispeech" / f"{split}.tar.gz"
    root = corpora / "librispeech" / "LibriSpeech" / split
    if not root.is_dir():
        raise SystemExit(f"{root} missing: extract {archive.name} in place first")
    refs: list[tuple[str, str]] = []
    paths: dict[str, Path] = {}
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            refs.append((uid, text))
            paths[uid] = trans.parent / f"{uid}.flac"
    with tarfile.open(archive) as tar:
        in_archive = sum(1 for m in tar if m.name.endswith(".flac"))
    on_disk = sum(1 for p in paths.values() if p.is_file())
    if not (len(refs) == on_disk == in_archive):
        raise SystemExit(
            f"{split}: {len(refs)} transcripts, {on_disk} flac files, {in_archive} in the archive"
        )

    def stream(wanted: set[str]) -> Iterator[tuple[str, Pcm]]:
        for uid, _ in refs:
            if uid in wanted:
                yield uid, _pcm(str(paths[uid]))

    return Corpus(refs, [archive], stream)


def parquet_corpus(
    files: list[Path], audio_col: str, ref_col: str, id_col: str | None, prefix: str
) -> Corpus:
    """A parquet corpus. A uid is ``prefix:<id>``, or ``prefix:<row>`` counted
    across the files in order when the corpus has no id column."""

    def uids(batch: dict[str, list[object]], first_row: int, n: int) -> list[str]:
        if id_col:
            return [f"{prefix}:{v}" for v in batch[id_col]]
        return [f"{prefix}:{first_row + i}" for i in range(n)]

    refs: list[tuple[str, str]] = []
    row = 0
    for f in files:
        t = pq.read_table(f, columns=[ref_col] + ([id_col] if id_col else [])).to_pydict()
        n = len(t[ref_col])
        refs += list(zip(uids(t, row, n), (str(x) for x in t[ref_col]), strict=True))
        row += n

    def stream(wanted: set[str]) -> Iterator[tuple[str, Pcm]]:
        row = 0
        for f in files:
            cols = [audio_col] + ([id_col] if id_col else [])
            for batch in pq.ParquetFile(f).iter_batches(batch_size=32, columns=cols):
                d = batch.to_pydict()
                for uid, audio in zip(uids(d, row, batch.num_rows), d[audio_col], strict=True):
                    if uid in wanted:
                        yield uid, _pcm(audio["bytes"])
                row += batch.num_rows

    return Corpus(refs, files, stream)


def open_corpus(name: str, corpora: Path) -> Corpus:
    """The named corpus; for multi_step, the seal is checked and applied first."""
    if name.startswith("librispeech-"):
        return librispeech(corpora, name.removeprefix("librispeech-"))
    if name == "peoples-speech":
        files = sorted((corpora / "peoples_speech").glob("clean-test-*.parquet"))
        return parquet_corpus(files, "audio", "text", "id", "ps")
    split = MQA_SPLITS[name]
    files = sorted((corpora / "spoken_mqa").glob(f"{split}-*.parquet"))
    c = parquet_corpus(files, "context", "context_transcript", None, split)
    if split != "multi_step_reasoning":
        return c
    index = json.loads((REPO / MQA_INDEX).read_text())
    refs = apply_seal(c.refs, (REPO / SEAL).read_text(), index, split, SEAL_SHA256)
    return Corpus(refs, c.files, c.stream)


def verify_files(corpora: Path, files: list[Path]) -> dict[str, str]:
    """sha256 of each corpus file, checked against ~/corpora/SHA256SUMS.

    Raises:
        LabelRunError: a file is missing from SHA256SUMS or does not match it.
    """
    actual = {str(f.relative_to(corpora)): file_sha256(f) for f in files}
    check_sha256(actual, parse_sha256sums((corpora / "SHA256SUMS").read_text()))
    return actual


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(name: str) -> dict[str, str]:
    """The local snapshot a model name resolves to, and its weights' sha256.

    Loaded with HF_HUB_OFFLINE set, so a run never re-resolves "main" upstream.
    """
    from faster_whisper.utils import download_model

    path = Path(download_model(name, local_files_only=True))
    return {
        "snapshot": path.name,
        "path": str(path),
        "model_bin_sha256": file_sha256(path / "model.bin"),
    }


def draw(refs: list[tuple[str, str]], limit: int, seed: int) -> list[tuple[str, str]]:
    """All rows, or a seeded draw of ``limit`` of them, kept in file order."""
    if limit <= 0 or limit >= len(refs):
        return refs
    picked = sorted(random.Random(seed).sample(range(len(refs)), limit))
    return [refs[i] for i in picked]


# ------------------------------------------------------------------ settings


def replay_settings(cfg: TwlConfig) -> ReplaySettings:
    """The live parameters, read from the deployed config and twl.stt_engine."""
    c = cfg.stt.commit
    return ReplaySettings(
        duty_max=c.duty_max,
        min_uncommitted_s=c.min_uncommitted_s,
        agreement_n=c.agreement_n,
        tail_guard_s=c.tail_guard_s,
        use_initial_prompt=c.use_initial_prompt,
        poll_s=POLL_S,
        min_decode_s=MIN_DECODE_S,
        stop_secs=cfg.vad.stop_secs,
    )


def revisions() -> dict[str, str]:
    """The corpus revisions src/scripts/fetch_corpora.sh pins."""
    text = (REPO / FETCH).read_text()
    out = dict(re.findall(r"^(MQA_REV|PS_REV)=([0-9a-f]{40})", text, re.M))
    if set(out) != {"MQA_REV", "PS_REV"}:
        raise SystemExit(f"revisions not found in {FETCH}")
    return out


def settings_record(cfg: TwlConfig, s: ReplaySettings) -> dict[str, object]:
    """Everything a label depends on. A resume with a different record stops."""
    script = hashlib.sha256((REPO / "src/scripts/commit_labels.py").read_bytes()).hexdigest()
    modules = {
        m: hashlib.sha256((REPO / f"src/twl/{m}.py").read_bytes()).hexdigest()
        for m in (
            "label_replay", "survival_labels", "stt_engine", "pacing", "commit", "seal",
            "label_io", "wer",
        )
    }  # fmt: skip
    return {
        "script_sha256": script,
        "module_sha256": modules,
        "packages": {p: metadata.version(p) for p in STACK},
        "models": {
            "hypothesis": model_identity(cfg.stt.partial_model or "tiny"),
            "final": model_identity(cfg.stt.model),
        },
        "replay": asdict(s),
        "hypothesis_model": cfg.stt.partial_model,
        "final_model": cfg.stt.model,
        "compute_type": cfg.stt.compute_type,
        "language": cfg.stt.language,
        "threads": {"hypothesis": cfg.stt.partial_cpu_threads, "final": cfg.stt.cpu_threads},
        "prompt_chars": PROMPT_CHARS,
        "clock": "simulated: 1x audio; each decode advances it by twl.pacing.modelled_decode_ms "
        f"({PARTIAL_INTERCEPT_MS} + {PARTIAL_MS_PER_S} ms per s of buffer), the cost the issuer "
        "is told; real decode wall time recorded, never used",
        "trimming_committer": "LocalAgreement-2",
        "gate": "off (duration_prior_s 0.0)",
        "alignment": {
            "window_s": WINDOW_S,
            "match_tol_s": MATCH_TOL_S,
            "gap_cost": GAP_COST,
            "sub_cost": SUB_COST,
        },
        "features": {
            "logprob": "log of faster-whisper's word probability: the mean decoder probability "
            "of the word's tokens in its word-timestamp pass",
            "seg_avg_logprob": "the segment's average token log-probability (decoder)",
            "agreed_prev": "the word matches a word of LA-2's stored remainder by the label's "
            "alignment rule (same token, centres within match_tol_s)",
            "la2_commit": "LA-2 committed this word on this offer",
            "partials_seen": "the hypothesis's 1-based index in the utterance",
            "since_commit_s": "issue time minus the end of the decode that last committed",
            "word_index": "index among the fresh words (punctuation-only tokens counted, not rows)",
        },
        "label_columns": list(LABEL_COLUMNS),
        "id_columns": list(ID_COLUMNS),
        "feature_columns": [
            "text_hash",
            "pos_from_end_s",
            "word_dur_s",
            "logprob",
            "seg_avg_logprob",
            "no_speech_prob",
            "partials_seen",
            "agreed_prev",
            "la2_commit",
            "buffer_s",
            "since_commit_s",
        ],
        "labels": {
            "label": "the word matches the final at its position under Word.key "
            "(lowercase, punctuation stripped), centres within match_tol_s",
            "label_wer_norm": "the same alignment on twl.wer.normalize's tokens "
            "(punctuation split, 0-10 spelled); which one is THE label is Ali's choice",
            "match_offset_s": "derived from label: NaN exactly when label is 0",
        },
        "departures": [
            "word timestamps on the hypotheses: LA-2 agrees and commits per word "
            "(live, deployed: per segment), so buffers, prompts and trim points differ",
            "final = base on the full utterance with word timestamps (live: the tail, "
            "no word times)",
            "turn time origin = the file's first sample (live: the VAD start less its pre-roll)",
            "the whole file is one turn: live, a pause of vad.stop_secs or more inside it ends "
            "the turn and the next one starts with the committer and issuer reset; each "
            "utterance records the internal silences of at least vad.stop_secs that "
            "faster-whisper's Silero finds at the live threshold, unpadded (vad_silences_ge_stop, "
            "vad_max_silence_s; not Pipecat's detector, no min_volume), and, separately, the "
            "gaps between the final's word timestamps (word_gaps_ge_stop, max_word_gap_s)",
            "endpoint = the final's last word end + vad.stop_secs (live: Silero's), with silence "
            "appended past the file's end; a hypothesis still decoding then is frozen, never "
            "offered, and has no rows (at_endpoint stop)",
        ],
    }


# ------------------------------------------------------------------ output

ROW_SCHEMA = pa.schema(
    [
        ("utt", pa.string()),
        ("hyp_index", pa.int32()),
        ("word_index", pa.int32()),
        ("text", pa.string()),
        ("key", pa.string()),
        ("text_hash", pa.int64()),
        ("pos_from_end_s", pa.float64()),
        ("word_dur_s", pa.float64()),
        ("logprob", pa.float64()),
        ("seg_avg_logprob", pa.float64()),
        ("no_speech_prob", pa.float64()),
        ("partials_seen", pa.int32()),
        ("agreed_prev", pa.int8()),
        ("la2_commit", pa.int8()),
        ("buffer_s", pa.float64()),
        ("since_commit_s", pa.float64()),
        ("issued_s", pa.float64()),
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("match_offset_s", pa.float64()),
        ("label", pa.int8()),
        ("label_wer_norm", pa.int8()),
    ]
)


def _with_meta(schema: pa.Schema, prov: dict[str, object]) -> pa.Schema:
    return schema.with_metadata({b"twl_provenance": json.dumps(prov).encode()})


def boot_ratio(num: list[float], den: list[float], seed: int = 0) -> str:
    """sum(num)/sum(den) with a percentile bootstrap CI over the units that have
    a denominator; "n/a" when none has. Formatted: "r [lo, hi], n=<units>"."""
    units = [(a, b) for a, b in zip(num, den, strict=True) if b > 0]
    if not units:
        return "n/a, n=0"
    point = sum(a for a, _ in units) / sum(b for _, b in units)
    rng = random.Random(seed)
    n = len(units)
    draws = []
    for _ in range(BOOT):
        pick = [units[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(a for a, _ in pick) / sum(b for _, b in pick))
    draws.sort()
    return (
        f"{point:.4f} [{draws[int(0.025 * BOOT)]:.4f}, {draws[int(0.975 * BOOT) - 1]:.4f}], n={n}"
    )


def vad_silences(pcm: Pcm, cfg: TwlConfig) -> tuple[int, float]:
    """Internal silences of at least vad.stop_secs by Silero, and the longest.

    faster-whisper's Silero, at the live confidence threshold and minimum
    silence, with no padding: the closest offline measure of where the live
    endpoint rule would end a turn inside the file. Not the live detector
    itself (Pipecat's Silero, which also applies min_volume).
    """
    opts = VadOptions(
        threshold=cfg.vad.confidence,
        min_silence_duration_ms=int(cfg.vad.stop_secs * 1000),
        speech_pad_ms=0,
    )
    spans = get_speech_timestamps(pcm, opts, sampling_rate=SR)
    gaps = [(b["start"] - a["end"]) / SR for a, b in itertools.pairwise(spans)]
    long = [g for g in gaps if g >= cfg.vad.stop_secs]
    return len(long), max(gaps, default=0.0)


# ------------------------------------------------------------------ main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--corpus", choices=CORPORA, required=True)
    p.add_argument("--corpora", type=Path, default=Path.home() / "corpora")
    p.add_argument("--limit", type=int, default=0, help="a seeded draw of this many; 0 = all")
    p.add_argument("--seed", type=int, default=20261118)
    p.add_argument("--tag", default="", help="suffix for the output names, e.g. pilot")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--approved", default="", help="Ali's GO for a whole-corpus run, quoted")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)  # one line per decode
    os.environ["HF_HUB_OFFLINE"] = "1"  # the pinned local snapshots, never "main" upstream
    os.chdir(REPO)
    try:
        run(args)
    except LabelRunError as e:
        raise SystemExit(f"commit_labels: {e}") from None


def run(args: argparse.Namespace) -> None:
    """One label run; every guard raises LabelRunError."""
    lock_fd = hold_run_lock(REPO / RUN_LOCK)  # held until exit: the fence, both ways
    cfg = load_config(CONFIG)
    settings = replay_settings(cfg)
    corpus = open_corpus(args.corpus, args.corpora)
    if any(not ref.strip() for _, ref in corpus.refs):
        raise LabelRunError(f"{args.corpus}: an empty reference; WER would drop its insertions")
    chosen = draw(corpus.refs, args.limit, args.seed)
    require_approval(len(chosen), args.approved)
    file_sha = verify_files(args.corpora, corpus.files)
    record = settings_record(cfg, settings)
    meta = build_run_meta(
        run_id=new_run_id("labels"),
        config_path=CONFIG,
        notes=f"commit_labels {args.corpus} {args.tag}".strip(),
        extra_software={p: metadata.version(p) for p in STACK},
    )
    refuse_dirty(meta.git_commit, args.out, REPO / RAW)

    stem = args.corpus + (f".{args.tag}" if args.tag else "")
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / f"{stem}.jsonl"
    header = {"header": record, "corpus": args.corpus, "seed": args.seed, "limit": args.limit}
    if ckpt.exists():
        cp = read_checkpoint(ckpt.read_text().splitlines(), header)
        if cp.truncated:
            log.warning("dropping a checkpoint line cut short by an earlier crash")
            lines = ckpt.read_text().splitlines()[:-1]
            ckpt.write_text("".join(line + "\n" for line in lines))
    else:
        ckpt.write_text(json.dumps(header) + "\n")
        cp = read_checkpoint([json.dumps(header)], header)
    invocation = {"run_meta": asdict(meta), "command": shlex.join(sys.argv)}
    with ckpt.open("a") as fh:
        fh.write(json.dumps({"invocation": invocation}) + "\n")
    invocations = [*cp.invocations, invocation]
    ref_of = dict(chosen)
    wanted = set(ref_of) - cp.done
    log.info("%s: %d utterances, %d already done", args.corpus, len(chosen), len(cp.done))

    cpus = list(cfg.stt.partial_cpus or cfg.stt.cpu_affinity)
    tiny_name = cfg.stt.partial_model or "tiny"
    tiny = load_pinned(tiny_name, cfg.stt.partial_cpu_threads, cpus, cfg.stt.compute_type)
    base = load_pinned(cfg.stt.model, cfg.stt.cpu_threads, cpus, cfg.stt.compute_type)
    lang = cfg.stt.language

    def decode_hyp(buf: Pcm, base_s: float, prompt: str) -> tuple[list[HypWord], int]:
        raw, wordless = transcribe_words_scored(
            tiny, buf, language=lang, base_s=base_s, initial_prompt=prompt
        )
        return [HypWord(Word(t, s0, s1), pr, alp, nsp) for t, s0, s1, pr, alp, nsp in raw], wordless

    t_run = time.perf_counter_ns()
    run_audio_s = 0.0
    run_decode_ms = 0.0
    with ckpt.open("a") as fh:
        for n, (uid, pcm) in enumerate(corpus.stream(wanted), 1):
            t0 = time.perf_counter_ns()
            raw_final, final_wordless = transcribe_words_scored(
                base, pcm, language=lang, base_s=0.0, initial_prompt=""
            )
            final_ms = (time.perf_counter_ns() - t0) / 1e6
            final = [Word(t, s0, s1) for t, s0, s1, *_ in raw_final]
            final_text = " ".join(w.text for w in final)
            gaps = [b.start_s - a.end_s for a, b in itertools.pairwise(final)]
            vad_n, vad_max = vad_silences(pcm, cfg)
            res = replay(pcm, final, decode_hyp, settings)
            reference = ref_of[uid]
            hyp_ms = sum(float(h["wall_ms"] or 0.0) for h in res.hyps)
            rec = {
                "utt": uid,
                "reference": reference,
                "final": final_text,
                "final_words": [
                    {"text": w.text, "start_s": w.start_s, "end_s": w.end_s} for w in final
                ],
                "final_wordless_segments": final_wordless,
                "wer": wer(reference, final_text),
                "ref_words": len(normalize(reference)),
                "dur_s": len(pcm) / SR,
                "endpoint_s": res.endpoint_s,
                "max_word_gap_s": max(gaps, default=0.0),
                "word_gaps_ge_stop": sum(1 for g in gaps if g >= settings.stop_secs),
                "vad_silences_ge_stop": vad_n,
                "vad_max_silence_s": vad_max,
                "hypotheses": res.hypotheses,
                "frozen": res.frozen,
                "empty_words": res.empty_words,
                "hyp_wordless_segments": res.wordless_segments,
                "hyps": res.hyps,
                "final_wall_ms": final_ms,
                "hyp_wall_ms_total": hyp_ms,
                "rows": res.rows,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            run_audio_s += len(pcm) / SR
            run_decode_ms += final_ms + hyp_ms
            if n % 25 == 0 or n == len(wanted):
                el = (time.perf_counter_ns() - t_run) / 1e9
                log.info("  %d/%d in %.0f s", n, len(wanted), el)
    run_wall_s = (time.perf_counter_ns() - t_run) / 1e9

    prov: dict[str, object] = {
        "invocations": invocations,
        "settings": record,
        "corpus": args.corpus,
        "role": ROLE[args.corpus],
        "corpus_files_sha256": file_sha,
        "corpus_revisions": revisions(),
        "corpora_sha256sums": str(args.corpora / "SHA256SUMS"),
        "seal_sha256": SEAL_SHA256 if args.corpus == "mqa-multi_step" else None,
        "seed": args.seed,
        "limit": args.limit,
        "pilot_max": PILOT_MAX,
        "approved": args.approved or None,
        "utterances_in_corpus": len(corpus.refs),
        "drawn_uids": [u for u, _ in chosen],
    }
    summarize(
        args,
        ckpt,
        ref_of,
        prov,
        len(corpus.refs),
        len(wanted),
        run_wall_s,
        run_audio_s,
        run_decode_ms,
    )
    os.close(lock_fd)


def summarize(
    args: argparse.Namespace,
    ckpt: Path,
    ref_of: dict[str, str],
    prov: dict[str, object],
    corpus_n: int,
    run_n: int,
    run_wall_s: float,
    run_audio_s: float,
    run_decode_ms: float,
) -> None:
    """Stream the checkpoint into the two parquet files and print LABELS and COST."""
    stem = ckpt.name.removesuffix(".jsonl")
    utts: list[dict[str, object]] = []
    pos: list[float] = []
    pos_norm: list[float] = []
    nwords: list[float] = []
    errs: list[float] = []
    refw: list[float] = []
    offsets: list[float] = []
    rows_path = args.out / f"{stem}.parquet"
    with ckpt.open() as fh, pq.ParquetWriter(rows_path, _with_meta(ROW_SCHEMA, prov)) as writer:
        fh.readline()
        for line in fh:
            rec = json.loads(line)
            if "invocation" in rec or rec["utt"] not in ref_of:
                continue
            rows = rec.pop("rows")
            for r in rows:
                r["utt"] = rec["utt"]
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=ROW_SCHEMA))
            utts.append(rec)
            pos.append(float(sum(int(r["label"]) for r in rows)))
            pos_norm.append(float(sum(int(r["label_wer_norm"]) for r in rows)))
            nwords.append(float(len(rows)))
            errs.append(float(rec["wer"]) * float(rec["ref_words"]))
            refw.append(float(rec["ref_words"]))
            offsets += [abs(float(r["match_offset_s"])) for r in rows if r["label"] == 1]
    missing = set(ref_of) - {str(u["utt"]) for u in utts}
    if missing:
        raise LabelRunError(
            f"{len(missing)} chosen utterances never labelled: {sorted(missing)[:3]}"
        )
    utt_table = pa.Table.from_pylist(utts)
    pq.write_table(
        utt_table.replace_schema_metadata(_with_meta(utt_table.schema, prov).metadata),
        args.out / f"{stem}.utterances.parquet",
    )

    def total(key: str) -> int:
        return sum(int(str(u[key])) for u in utts)

    words = int(sum(nwords))
    no_rows = sum(1 for n in nwords if n == 0)
    off_med = statistics.median(offsets) if offsets else float("nan")
    off_p95 = sorted(offsets)[int(0.95 * (len(offsets) - 1))] if offsets else float("nan")
    with_vad = sum(1 for u in utts if int(str(u["vad_silences_ge_stop"])) > 0)
    with_gap = sum(1 for u in utts if int(str(u["word_gaps_ge_stop"])) > 0)
    name = args.corpus + (f" {args.tag}" if args.tag else "")
    print(
        f"LABELS {name}: utterances {len(utts)} ({no_rows} without rows); partials "
        f"{total('hypotheses')} (frozen {total('frozen')}); words {words} (punctuation-only "
        f"tokens, not rows: {total('empty_words')}); positive {int(sum(pos))}/{words} = "
        f"{boot_ratio(pos, nwords)}; under WER tokens {int(sum(pos_norm))}/{words} = "
        f"{boot_ratio(pos_norm, nwords)} (bootstrap over utterances with rows); final WER "
        f"{boot_ratio(errs, refw)} over {int(sum(refw))} reference words; |matched centre "
        f"offset| median {off_med:.3f} s, p95 {off_p95:.3f} s (n={len(offsets)}); utterances "
        f"with an internal silence >= vad.stop_secs: Silero {with_vad}/{len(utts)}, word "
        f"timestamps {with_gap}/{len(utts)}; wordless segments: final "
        f"{total('final_wordless_segments')}, hypotheses {total('hyp_wordless_segments')}"
    )
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    proj = f"{run_wall_s / run_n * corpus_n / 3600:.2f} h" if run_n else "n/a (nothing run here)"
    print(
        f"COST {name}: this run {run_n} utterances, audio {run_audio_s / 3600:.3f} h, wall "
        f"{run_wall_s:.0f} s, decode {run_decode_ms / 1000:.0f} s; whole corpus ({corpus_n} "
        f"utterances) at this run's wall per utterance: {proj}; peak RSS {rss_mb:.0f} MB"
    )


if __name__ == "__main__":
    main()
