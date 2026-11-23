"""Audit: which live turns did the pre-roll defect reach? By CONTENT (Ali, 2026-11-20).

THE DEFECT (fixed in 17a50c7). stt.py fills a PREROLL_S (0.6 s, RING = 30
delivered chunks) ring only while no turn is open, and never emptied it at a
turn's close. A turn opening fewer than RING chunks of DELIVERED audio after
the previous close therefore started with audio from before the previous
open. The criterion is in delivered chunks, not wall time: during the
playback gate's wait no audio is delivered.

THE RECORD. Each final saves its buffer as segments/turn_NNN.wav: the
pre-roll ring followed by every chunk delivered while the turn was open, cut
at the committed end (committed_end_s, 0 unless a commit arm committed). NNN
is the turn current when the final completed, so a later final could
OVERWRITE the file after its record was written: a file whose length is not
the record's stt_audio_s (set by the same final) holds another buffer, and
the record's own head is unknown. (An overwrite by a buffer of exactly the
same length would pass; PREROLL OVERWRITTEN prints the smallest length
difference among those found.) The
played files and playback_timeline.json give the delivered stream chunk by
chunk; a chunk is compared after the pipeline's own conversion (int16 to
float / 32768 at stt.py's chunk intake, clip * 32767 to int16 at the save).

1. LOCATE each saved segment: every non-zero chunk of its BODY (buffer chunk
   RING and later) votes for the delivered index of the turn's close. The
   position is taken when at least LOCATE_SHARE of the voting chunks agree; of
   the positions that do, the first whose open is at or after the previous
   located close (records in turn order: a run may replay a file). The open
   is close - buffer chunks + RING. A body where not every chunk agrees is
   NON-CONTIGUOUS.
2. CONTENT (committed_end_s == 0: the head is saved): the 30 head chunks
   against the delivered chunks just before the open, which is what a fresh
   ring holds. Any chunk differing: STALE; stale_ms is 20 ms times the head
   chunks before the trailing run that matches.
3. POSITION (committed_end_s > 0: the head was cut off with the commit):
   the gap from the previous located close to this open, in delivered
   chunks, the previous close being any saved buffer's, an overwritten
   one's included (located from its last chunks; kept only when one
   position qualifies). Fresh when the gap is at least RING chunks AND no
   unsaved close can hide in it: no chunk louder than QUIET_DBFS (RMS) in
   [previous close, open - stop_secs), with stop_secs from a run config whose
   sha256 is the run's config_hash. A hidden turn needs speech to open; its
   final can be lost to an overwrite. QUIET_DBFS is set far below anything
   the VAD's volume gate passes (pipecat's volume reads about 0.5, against
   min_volume 0.6, already at -60 dBFS, though its smoothing can lag after
   loud audio); POSITION_MAX prints the loudest gap among the turns it
   clears. Otherwise UNVERIFIABLE. CROSSCHECK prints this rule against the content check on
   every turn where both apply; SELFCHECK cuts head-saved segments as a
   commit would (int(ce * 16000)) and requires the same position.

Classes: fresh, stale, unverifiable (segment overwritten, position not
decidable, non-contiguous body, segment not locatable), unchecked (no
timeline, a played file missing, no saved segment). The open/close/gap fields
assume a full ring: they are off where the ring was short (a run's first
turn, and runs after 17a50c7, whose records log preroll_s).

This audit names turns; it does not apply any scorer's filter.
preroll_exposure.py does, per pass.

Run from the repository root with PYTHONPATH=src:

    preroll_audit.py [--raw results/raw/reactive] [--out FILE.jsonl] [--log FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

from twl.stt_engine import PREROLL_S
from twl.transport import CHUNK_MS

SR = 16000
CH = SR * CHUNK_MS // 1000  # samples per delivered chunk
RING = round(PREROLL_S * 1000) // CHUNK_MS  # pre-roll chunks
ZERO = bytes(2 * CH)
LOCATE_SHARE = 0.95
QUIET_DBFS = -90.0
LENGTH_TOL_S = 0.001  # stt_audio_s is rounded to 1 ms

I16 = npt.NDArray[np.int16]


def pipeline_int16(a: I16) -> I16:
    """What the saved segment holds for delivered int16 samples ``a``."""
    f = a.astype(np.float32) / 32768.0  # stt.py chunk intake
    return (np.clip(f, -1.0, 1.0) * 32767).astype(np.int16)  # stt.py _save_segment


def read_wav(path: str | Path) -> I16:
    with wave.open(str(path)) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SR):
            raise ValueError(f"{path}: not 16 kHz mono int16")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def chunk(a: I16, i: int) -> bytes:
    return a[i * CH : (i + 1) * CH].tobytes()


@dataclass
class Stream:
    """The delivered stream: chunk bytes by delivered index (absent = silence)."""

    chunks: dict[int, bytes] = field(default_factory=dict)
    where: dict[bytes, list[int]] = field(default_factory=lambda: defaultdict(list))

    def at(self, g: int) -> bytes:
        return self.chunks.get(g, ZERO)


def delivered_stream(timeline: list[list[object]]) -> Stream:
    """From playback_timeline.json: [path, first delivered chunk, chunks] per file."""
    s = Stream()
    for path, start, n in timeline:
        a = pipeline_int16(read_wav(str(path)))
        a = np.concatenate([a, np.zeros(-len(a) % CH, np.int16)])
        if len(a) // CH != int(str(n)):
            raise ValueError(f"{path}: {len(a) // CH} chunks, timeline says {n}")
        for j in range(int(str(n))):
            b = chunk(a, j)
            g = int(str(start)) + j
            s.chunks[g] = b
            if b != ZERO:
                s.where[b].append(g)
    return s


def segment_offset(committed_end_s: float, n: int) -> int:
    """The buffer sample where the saved segment starts. The buffer is whole
    chunks, so the offset is congruent to -n modulo CH; of those offsets, the
    nearest to committed_end_s * SR."""
    est = committed_end_s * SR
    base = int(est) - ((int(est) + n) % CH)
    return min((base, base + CH), key=lambda o: abs(o - est))


@dataclass
class Located:
    nbuf: int  # buffer chunks (pre-roll + open)
    close: int  # delivered index just past the turn's last chunk
    contiguous: bool  # every voting body chunk agrees

    @property
    def open(self) -> int:
        return self.close - self.nbuf + RING


def locate(seg: I16, committed_end_s: float, s: Stream, after: int) -> Located | str:
    """Where the segment's body sits in the delivered stream, opening at or after ``after``."""
    o = segment_offset(committed_end_s, len(seg)) if committed_end_s > 0 else 0
    if (o + len(seg)) % CH:
        return "misaligned"
    nbuf = (o + len(seg)) // CH
    votes: Counter[int] = Counter()
    voters = 0
    for b in range(max(-(-o // CH), RING), nbuf):
        c = seg[b * CH - o : (b + 1) * CH - o].tobytes()
        if c == ZERO:
            continue
        voters += 1
        for g in s.where.get(c, ()):
            votes[g + nbuf - b] += 1
    ok = sorted(g for g, v in votes.items() if voters and v >= LOCATE_SHARE * voters)
    later = [g for g in ok if g - nbuf + RING >= after]
    if not later:
        return "unlocated"
    return Located(nbuf, later[0], votes[later[0]] == voters)


def head_stale_ms(seg: I16, s: Stream, opened: int) -> int:
    """0 if the 30 head chunks are what a fresh ring holds; else the ms before
    the trailing run of head chunks that match it."""
    fresh = 0
    for i in range(RING - 1, -1, -1):
        if chunk(seg, i) != s.at(opened - RING + i):
            break
        fresh += 1
    return (RING - fresh) * CHUNK_MS


def loudest_dbfs(s: Stream, a: int, b: int) -> float:
    """The loudest chunk in [a, b), RMS in dBFS; -inf for none or silence."""
    best = 0.0
    for g in range(a, b):
        x = np.frombuffer(s.at(g), dtype=np.int16).astype(np.float64)
        best = max(best, float(np.sqrt(np.mean(x * x))))
    return 20 * float(np.log10(best / 32768)) if best > 0 else float("-inf")


_vad_cache: dict[tuple[str, str, str], tuple[int, bool]] = {}
NOTES: list[str] = []  # printed with the summary
OVERWRITE_GAPS: list[float] = []  # |file - record| length of each overwritten file, in chunks


def run_vad(commit: str, config_path: str, config_hash: str) -> tuple[int, bool]:
    """(stop_secs in chunks, verified): the run's config at its commit, or on
    disk when the repository never had it; verified when its sha256 is the
    run's recorded config_hash. The VadConfig default where it sets none."""
    key = (commit, config_path, config_hash)
    if key not in _vad_cache:

        def show(path: str) -> bytes:
            return subprocess.run(
                ["git", "show", f"{commit}:{path}"], capture_output=True, check=True
            ).stdout

        try:
            raw = show(config_path)
        except subprocess.CalledProcessError:  # a config outside the repository
            raw = Path(config_path).read_bytes() if Path(config_path).exists() else b""
        verified = bool(raw) and hashlib.sha256(raw).hexdigest() == config_hash
        if not verified:
            NOTES.append(
                f"# vad: {config_path} at {commit[:12]} does not match the run's config_hash: "
                "its position verdicts are unverifiable"
            )
        vad = (yaml.safe_load(raw.decode()) if raw else None) or {}
        vad = vad.get("vad") or {}
        cls = show("src/twl/config.py").decode()
        cls = cls[cls.index("class VadConfig") :]
        if "stop_secs" in vad:
            stop = float(vad["stop_secs"])
        else:
            m = re.search(r"^\s+stop_secs: float = ([0-9.]+)", cls, re.M)
            if m is None:
                raise ValueError(f"{commit}: no VadConfig.stop_secs")
            stop = float(m.group(1))
        _vad_cache[key] = (round(stop * 1000) // CHUNK_MS, verified)
    return _vad_cache[key]


def other_closes(other: list[tuple[I16, int]], s: Stream) -> set[int]:
    """The closes of overwritten buffers: located from the chunks after the
    first RING (the head may be stale), at or after the chain's close at
    their record, and kept only when one position qualifies."""
    out = set()
    for seg, after in other:
        o = -len(seg) % CH  # the cut is unknown; any offset congruent to it
        found = locate(seg, o / SR, s, after) if o else locate(seg, 0.0, s, after)
        if isinstance(found, Located):
            again = (
                locate(seg, o / SR, s, found.close + 1 - found.nbuf + RING)
                if o
                else (locate(seg, 0.0, s, found.close + 1 - found.nbuf + RING))
            )
            if not isinstance(again, Located):  # no second position: unambiguous
                out.add(found.close)
    return out


def turn_records(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    meta: dict[str, object] = {}
    recs = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "turn_record":
                recs.append(r)
            elif not meta:
                meta = r
    return meta, sorted(recs, key=lambda r: int(str(r["turn"])))


def classify_run(
    run: Path,
) -> tuple[list[dict[str, object]], Counter[tuple[str, str, str]], Counter[str]]:
    """One row per turn record; the content-vs-position cross-check counts; and
    the self-check of locating a committed cut."""
    meta, recs = turn_records(run / "turns.jsonl")
    commit = str(meta.get("git_commit", ""))

    def row(r: dict[str, object], cls: str, why: str, **extra: object) -> dict[str, object]:
        return {
            "run": run.name,
            "turn": int(str(r["turn"])),
            "valid": bool(r.get("valid")),
            "warmup": bool(r.get("warmup")),
            "washout": bool(r.get("washout")),
            "invalid_reason": str(r.get("invalid_reason") or ""),
            "build": commit,
            "cls": cls,
            "why": why,
            **extra,
        }

    xcheck: Counter[tuple[str, str, str]] = Counter()
    selfcheck: Counter[str] = Counter()
    tl = run / "playback_timeline.json"
    if not tl.exists():
        return [row(r, "unchecked", "no timeline") for r in recs], xcheck, selfcheck
    timeline = json.loads(tl.read_text())
    missing = [p for p, _, _ in timeline if not Path(str(p)).exists()]
    if missing:
        return [row(r, "unchecked", "played file missing") for r in recs], xcheck, selfcheck
    s = delivered_stream(timeline)
    stop, vad_ok = run_vad(
        commit.removesuffix("-dirty"), str(meta["config_path"]), str(meta.get("config_hash"))
    )
    # Pass 1: each record's own saved buffer, located in turn order; an
    # overwritten file is kept aside with the chain's position at its record.
    own: dict[int, tuple[I16, float, Located]] = {}
    other: list[tuple[I16, int]] = []
    pre: dict[int, tuple[str, str]] = {}
    last_close = 0
    for i, r in enumerate(recs):
        seg_path = str(r.get("segment_wav") or "")
        if not seg_path or not Path(seg_path).exists():
            pre[i] = ("unchecked", "no saved segment")
            continue
        seg = read_wav(seg_path)  # already through pipeline_int16
        off_s = abs(len(seg) / SR - float(str(r.get("stt_audio_s", -1))))
        if off_s > LENGTH_TOL_S:
            pre[i] = ("unverifiable", "segment overwritten by a later final")
            OVERWRITE_GAPS.append(off_s * 1000 / CHUNK_MS)
            other.append((seg, last_close))
            continue
        ce = float(str(r.get("committed_end_s") or 0.0))
        loc = locate(seg, ce, s, last_close)
        if isinstance(loc, str):
            pre[i] = ("unverifiable", f"segment {loc}")
            continue
        own[i] = (seg, ce, loc)
        last_close = loc.close
    # Pass 2: every close a saved buffer shows, an overwritten one's included
    # (its close needs no commit offset: it is its last chunk). Taken only
    # where unambiguous.
    own_closes = {loc.close for _, _, loc in own.values()}
    closes = sorted(own_closes | other_closes(other, s))
    out = []
    for i, r in enumerate(recs):
        if i in pre:
            out.append(row(r, *pre[i]))
            continue
        seg, ce, loc = own[i]
        prev_close = max((c for c in closes if c <= loc.open), default=0)
        gap = loc.open - prev_close
        loudest = loudest_dbfs(s, prev_close, loc.open - stop)
        if gap < RING:
            reason = "gap under the pre-roll"
        elif loudest > QUIET_DBFS:
            reason = "sound in the gap"
        elif not vad_ok:
            reason = "run config unverified"
        else:
            reason = "position"
        by_position = "fresh" if reason == "position" else "unverifiable"
        extra = {"open": loc.open, "close": loc.close, "gap_chunks": gap, "gap_dbfs": loudest}
        if ce == 0:
            ms = head_stale_ms(seg, s, loc.open)
            xcheck[("stale" if ms else "fresh", by_position, reason)] += 1
            before = max((c for c in own_closes if c <= loc.open), default=0)
            if before != prev_close:  # the verdict leans on an overwritten buffer's close
                xcheck[("stale" if ms else "fresh", by_position, "via an overwritten close")] += 1
            out.append(row(r, "stale" if ms else "fresh", "content", stale_ms=ms, **extra))
            if loc.contiguous and loc.nbuf - RING >= 60:  # a commit's cut, as stt.py makes it
                cut_s = (RING + (loc.nbuf - RING) // 2) * CHUNK_MS / 1000 + 0.0071
                cut = locate(seg[int(cut_s * SR) :], round(cut_s, 3), s, prev_close)
                selfcheck[
                    "unlocated"
                    if isinstance(cut, str)
                    else ("same" if (cut.open, cut.close) == (loc.open, loc.close) else "different")
                ] += 1
        elif not loc.contiguous:
            out.append(row(r, "unverifiable", "non-contiguous body", **extra))
        else:
            out.append(row(r, by_position, reason, **extra))
    return out, xcheck, selfcheck


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw", type=Path, default=Path("results/raw/reactive"))
    p.add_argument("--out", type=Path, default=None, help="every turn's class, JSONL")
    p.add_argument("--log", type=Path, default=None, help="also write the summary here")
    args = p.parse_args()
    build = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if (args.log or args.out) and build.endswith("-dirty"):
        raise SystemExit(f"build {build} is dirty: a logged audit needs a clean tree")
    lines = [
        f"# preroll_audit.py build {build}; RING {RING} chunks of {CHUNK_MS} ms; "
        f"locate share {LOCATE_SHARE}; quiet at or below {QUIET_DBFS} dBFS RMS"
    ]
    rows: list[dict[str, object]] = []
    xcheck: Counter[tuple[str, str, str]] = Counter()
    selfcheck: Counter[str] = Counter()
    runs = sorted(d for d in args.raw.glob("reactive-*") if (d / "turns.jsonl").exists())
    for d in runs:
        r, x, sc = classify_run(d)
        rows += r
        xcheck += x
        selfcheck += sc
    lines += NOTES
    by: Counter[tuple[str, str]] = Counter((str(r["cls"]), str(r["why"])) for r in rows)
    lines.append(f"PREROLL AUDIT: {len(runs)} runs, {len(rows)} turn records")
    for (cls, why), n in sorted(by.items()):
        v = sum(1 for r in rows if (r["cls"], r["why"]) == (cls, why) and r["valid"])
        lines.append(f"PREROLL CLASS {cls} ({why}): {n} records, of them valid {v}")

    via = "via an overwritten close"

    def xc(content: str, position: str, only: str | None = None) -> int:
        return sum(
            n
            for (c, p, why), n in xcheck.items()
            if (c, p) == (content, position) and (why == only if only else why != via)
        )

    split = ", ".join(
        f"{why} {n}" for (c, _, why), n in sorted(xcheck.items()) if c == "stale" and why != via
    )
    total = sum(xc(c, p) for c in ("fresh", "stale") for p in ("fresh", "unverifiable"))
    lines.append(
        "PREROLL CROSSCHECK on turns with the head saved (content vs position rule): "
        f"content fresh, position fresh {xc('fresh', 'fresh')}; content stale, position "
        f"unverifiable {xc('stale', 'unverifiable')} ({split}); content fresh, position "
        f"unverifiable {xc('fresh', 'unverifiable')}; content stale, position fresh "
        f"{xc('stale', 'fresh')} (must be 0); of {total}"
    )
    lines.append(
        "PREROLL CROSSCHECK, the turns whose position verdict leans on an overwritten "
        f"buffer's close: content fresh, position fresh {xc('fresh', 'fresh', via)}; content "
        f"fresh, position unverifiable {xc('fresh', 'unverifiable', via)}; content stale, "
        f"position fresh {xc('stale', 'fresh', via)} (must be 0); content stale, position "
        f"unverifiable {xc('stale', 'unverifiable', via)}"
    )
    lines.append(
        f"PREROLL OVERWRITTEN: {len(OVERWRITE_GAPS)} files whose length is not their record's; "
        f"the smallest difference {min(OVERWRITE_GAPS, default=float('nan')):.1f} chunks"
    )
    cleared = [float(str(r["gap_dbfs"])) for r in rows if r["why"] == "position"]
    lines.append(
        f"PREROLL POSITION_MAX: the loudest gap among the {len(cleared)} turns cleared by "
        f"position: {max(cleared, default=float('-inf')):.1f} dBFS (threshold {QUIET_DBFS})"
    )
    lines.append(
        "PREROLL SELFCHECK, head-saved segments cut mid-buffer as a commit cuts them: "
        f"same position {selfcheck['same']}; different {selfcheck['different']} (must be 0); "
        f"unlocated {selfcheck['unlocated']}"
    )
    flagged = [r for r in rows if r["cls"] in ("stale", "unverifiable") and r["valid"]]
    per_run: dict[str, list[str]] = defaultdict(list)
    for r in flagged:
        tag = f"t{r['turn']} {r['cls']}"
        tag += f" {r['stale_ms']} ms" if r["cls"] == "stale" else f" ({r['why']})"
        per_run[str(r["run"])].append(tag)
    for run, tags in per_run.items():
        lines.append(f"PREROLL VALID FLAGGED {run}: {len(tags)}: {', '.join(tags)}")
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
