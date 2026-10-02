"""Offline smoke for COMMIT-WL: does it cost WER, and which endpoint rule wins?

Replays saved segments through the real engines with no pipeline, so the
comparison is paired on byte-identical audio and nothing is timed against a
live VAD. Three questions, all of which set a configuration choice:

  1. WER. COMMIT-WL's transcript against the one-shot baseline's, both against
     the manifest. The acceptance bar is REACTIVE + 0.02 (v3 §2.6).
  2. race vs wait. How much audio the final is handed, and what that costs, when
     the endpoint discards the hypothesis in flight versus waiting for it.
  3. word_timestamps. What asking for word times costs the hypothesis decode.
     Over 15 % of the decode and the fallback is segment-granularity commits.

Hypotheses are issued by the real self-paced rule against a simulated clock:
audio arrives at 1x, the rule decides when it would have issued, and the decode
is then run for real. That keeps the pacing honest without a live pipeline.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import time
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

from twl.commit import LocalAgreementCommitter, Word
from twl.pacing import SelfPacedIssuer
from twl.twotier import Job, Span, TwoTierListener, choose_job
from twl.wer import wer


def read_wav(path: Path) -> tuple[npt.NDArray[np.float32], float]:
    with wave.open(str(path)) as w:
        n, rate = w.getnframes(), w.getframerate()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, n / rate


def words_of(segments: object, base_s: float, want_words: bool) -> list[Word]:
    out: list[Word] = []
    for seg in segments:  # type: ignore[attr-defined]
        if seg.no_speech_prob >= 0.6:
            continue
        if want_words and getattr(seg, "words", None):
            out.extend(
                Word(w.word.strip(), base_s + w.start, base_s + w.end)
                for w in seg.words
                if w.word.strip()
            )
        elif seg.text.strip():
            out.append(Word(seg.text.strip(), base_s + seg.start, base_s + seg.end))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    p.add_argument("--duty-max", type=float, default=0.6)
    p.add_argument("--agreement-n", type=int, default=2)
    p.add_argument("--tail-guard-s", type=float, default=0.3)
    p.add_argument("--hypothesis-model", default="tiny", choices=("tiny", "base"))
    p.add_argument(
        "--two-tier",
        action="store_true",
        help="tiny picks the boundaries, base re-decodes each committed span "
        "and its text is the transcript",
    )
    p.add_argument("--out", type=Path, default=Path("results/raw/commit_smoke.json"))
    args = p.parse_args()

    from faster_whisper import WhisperModel

    manifest = args.wav_dir / "manifest.csv"
    with open(manifest, encoding="utf-8") as fh:
        refs = {row["file"]: row["transcript"] for row in csv.DictReader(fh)}
    wavs = sorted(args.wav_dir.glob("*.wav"))
    if not wavs:
        print(f"no wavs in {args.wav_dir}")
        return

    base = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=3)
    # The hypothesis engine. base as hypothesis means ONE model resident
    # instead of two, which is 134 MB of the recogniser's footprint back —
    # and the occupancy tax says footprint is not free on this device.
    tiny = (
        base
        if args.hypothesis_model == "base"
        else WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=3)
    )
    for m in ({id(tiny): tiny, id(base): base}).values():
        m.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)

    def decode(model: object, audio: npt.NDArray[np.float32], **kw: object) -> object:
        segs, _info = model.transcribe(  # type: ignore[attr-defined]
            audio,
            language="en",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
            **kw,
        )
        return list(segs)

    rows = []
    for wav in wavs:
        pcm, dur = read_wav(wav)
        ref = refs.get(wav.name, "")
        rate = 16000

        t0 = time.perf_counter_ns()
        baseline = " ".join(
            s.text.strip()
            for s in decode(base, pcm)
            if s.text.strip()  # type: ignore[union-attr]
        ).strip()
        baseline_ms = (time.perf_counter_ns() - t0) / 1e6

        per_mode = {}
        modes = (True,) if args.two_tier else (True, False)
        for want_words in modes:
            committer = LocalAgreementCommitter(
                agreement_n=args.agreement_n, tail_guard_s=args.tail_guard_s
            )
            issuer = SelfPacedIssuer(duty_max=args.duty_max)
            two = TwoTierListener()
            now_s, hyp_ms, n_hyp = 0.0, [], 0
            issue_at: list[float] = []
            span_ms: list[float] = []
            spans_done, spans_queued = 0, 0
            spans_before_endpoint = 0
            idle_ms = 1e9
            while now_s < dur:
                d = issuer.decide(
                    in_flight=False,
                    uncommitted_s=now_s - committer.committed_end_s,
                    idle_ms=idle_ms,
                    buffer_s=now_s - committer.committed_end_s,
                )
                job = (
                    choose_job(
                        in_flight=False,
                        duty_satisfied=d.reason != "duty",
                        has_pending_span=two.next_span() is not None,
                        hypothesis_wanted=d.issue,
                    )
                    if args.two_tier
                    else (Job.TINY_HYPOTHESIS if d.issue else Job.NONE)
                )
                if job is Job.NONE:
                    now_s += 0.05
                    idle_ms += 50
                    continue
                if job is Job.BASE_SPAN:
                    sp = two.next_span()
                    assert sp is not None
                    seg_buf = pcm[int(sp.start_s * rate) : int(sp.end_s * rate)]
                    t1 = time.perf_counter_ns()
                    segs = decode(base, seg_buf, initial_prompt=two.prompt() or None)
                    ms = (time.perf_counter_ns() - t1) / 1e6
                    text = " ".join(
                        x.text.strip()
                        for x in segs
                        if x.text.strip()  # type: ignore[union-attr]
                    ).strip()
                    two.complete(sp, text)
                    spans_done += 1
                    span_ms.append(ms)
                    if now_s + ms / 1000.0 <= dur:
                        spans_before_endpoint += 1
                    issuer.note_decode(ms)
                    now_s += (ms + issuer.required_idle_ms(1.0)) / 1000.0
                    idle_ms = 0.0
                    continue
                start = committer.committed_end_s
                buf = pcm[int(start * rate) : int(now_s * rate)]
                if len(buf) < int(0.2 * rate):
                    now_s += 0.05
                    continue
                issue_at.append(now_s)
                t1 = time.perf_counter_ns()
                segs = decode(
                    tiny,
                    buf,
                    word_timestamps=want_words,
                    initial_prompt=(two.prompt() if args.two_tier else committer.prompt()) or None,
                )
                ms = (time.perf_counter_ns() - t1) / 1e6
                hyp_ms.append(ms)
                n_hyp += 1
                before = committer.committed_end_s
                kept = committer.offer(words_of(segs, start, want_words), buffer_end_s=now_s)
                if args.two_tier and kept:
                    two.enqueue(
                        Span(before, committer.committed_end_s, " ".join(w.text for w in kept))
                    )
                    spans_queued += 1
                issuer.note_decode(ms)
                now_s += (ms + issuer.required_idle_ms(now_s - committer.committed_end_s)) / 1000.0
                idle_ms = 0.0
            tail_from = two.base_committed_end_s if args.two_tier else committer.committed_end_s
            race_from = tail_from
            in_flight_ms = max(0.0, (now_s - dur) * 1000.0)
            t2 = time.perf_counter_ns()
            tail_segs = decode(base, pcm[int(tail_from * rate) :])
            tail_ms = (time.perf_counter_ns() - t2) / 1e6
            tail_text = " ".join(
                x.text.strip()
                for x in tail_segs
                if x.text.strip()  # type: ignore[union-attr]
            ).strip()
            committed = two.text() if args.two_tier else committer.committed_text()
            final = f"{committed} {tail_text}".strip()
            per_mode[want_words] = {
                "text": final,
                "committed_end_s": float(tail_from),
                "tail_s": float(dur - tail_from),
                "tail_ms": tail_ms,
                "hyp_ms": hyp_ms,
                "n_hyp": n_hyp,
                "wait_extra_ms": in_flight_ms,
                "cadence_s": [float(b - a) for a, b in itertools.pairwise(issue_at)],
                "spans_queued": spans_queued,
                "spans_done": spans_done,
                "spans_before_endpoint": spans_before_endpoint,
                "span_ms": span_ms,
                # ACHIEVED duty over the whole listening window: every decode
                # either engine ran, over the audio it ran during. This is the
                # quantity the VAD starvation was traced to, and it must hold
                # whichever engine spends it.
                "achieved_duty": float(
                    (sum(hyp_ms) + sum(span_ms)) / (now_s * 1000.0) if now_s > 0 else -1.0
                ),
                # CAST AT THE BOUNDARY. Whisper's timestamps are numpy
                # scalars, so every value derived from them is one too, and
                # json refuses them with a message that names the Python type.
                "committed": bool(committer.committed_end_s > 0),
                "tiny_committed_end_s": float(committer.committed_end_s),
                "base_covered_frac": float(
                    two.base_committed_end_s / committer.committed_end_s
                    if committer.committed_end_s > 0
                    else 0.0
                ),
            }
            del race_from
        if args.two_tier:
            per_mode[False] = per_mode[True]

        rows.append(
            {
                "wav": wav.name,
                "audio_s": dur,
                "ref": ref,
                "baseline_text": baseline,
                "baseline_ms": baseline_ms,
                "baseline_wer": wer(ref, baseline) if ref else -1.0,
                "words": per_mode[True],
                "segments": per_mode[False],
                "words_wer": wer(ref, per_mode[True]["text"]) if ref else -1.0,
                "segments_wer": wer(ref, per_mode[False]["text"]) if ref else -1.0,
            }
        )
        print(
            f"  {wav.name:>14} {dur:5.1f}s  base {baseline_ms:6.0f} ms  "
            f"tail {per_mode[True]['tail_ms']:6.0f} ms over "
            f"{per_mode[True]['tail_s']:4.1f}s  hyp x{per_mode[True]['n_hyp']}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1) + "\n")

    def med(key: str) -> float:
        return statistics.median([float(r[key]) for r in rows])  # type: ignore[arg-type]

    print(
        f"\nSMOKE on {len(rows)} segments, duty_max {args.duty_max}, "
        f"agreement_n {args.agreement_n}, tail_guard {args.tail_guard_s}"
    )
    print(f"  baseline WER      {med('baseline_wer'):.3f}")
    print(f"  COMMIT-WL WER     {med('words_wer'):.3f}   (word timestamps)")
    print(f"  COMMIT-WL WER     {med('segments_wer'):.3f}   (segment granularity)")
    delta = med("words_wer") - med("baseline_wer")
    print(
        f"  dWER              {delta:+.3f}   bar is +0.020 -> {'PASS' if delta <= 0.02 else 'FAIL'}"
    )

    w_ms = [x for r in rows for x in r["words"]["hyp_ms"]]  # type: ignore[index]
    s_ms = [x for r in rows for x in r["segments"]["hyp_ms"]]  # type: ignore[index]
    if w_ms and s_ms:
        cost = (statistics.median(w_ms) - statistics.median(s_ms)) / statistics.median(s_ms)
        print(
            f"\n  word_timestamps cost {cost:+.1%} of the hypothesis decode "
            f"({statistics.median(w_ms):.0f} vs {statistics.median(s_ms):.0f} ms)"
        )
        print(f"  bar is 15% -> {'keep word timestamps' if cost <= 0.15 else 'fall back'}")

    waits = [float(r["words"]["wait_extra_ms"]) for r in rows]  # type: ignore[index]
    print("\n  RACE vs WAIT: waiting for the hypothesis in flight would add a")
    print(f"  median {statistics.median(waits):.0f} ms before the tail final starts")
    print(f"  (max {max(waits):.0f} ms over {len(waits)} segments). It buys a shorter")
    print("  tail only when that hypothesis commits, which needs a SECOND")
    print("  agreeing hypothesis that by definition has not been issued.")

    cad = [c for r in rows for c in r["segments"]["cadence_s"]]  # type: ignore[index]
    nhyp = [float(r["segments"]["n_hyp"]) for r in rows]  # type: ignore[index]
    if cad:
        print(
            f"\n  achieved cadence {statistics.median(cad):.2f} s "
            f"(min {min(cad):.2f}, max {max(cad):.2f}, n={len(cad)}); "
            f"{statistics.median(nhyp):.0f} hypotheses per utterance"
        )
        print(
            f"  distinct cadences to 0.1 s: {len({round(c, 1) for c in cad})} "
            f"— the controller-varies check wants this > 1"
        )

    print("\n  PER-REP DETAIL (this rep):")
    nrows = len(rows)
    committed_on = sum(1 for r in rows if r["words"]["committed"])  # type: ignore[index]
    duty = [float(r["words"]["achieved_duty"]) for r in rows]  # type: ignore[index]
    print(f"    commits on {committed_on}/{nrows} items")
    print(
        f"    achieved duty (tiny+base) median {statistics.median(duty):.2f} "
        f"(max {max(duty):.2f}); duty_max {args.duty_max}"
    )
    if args.two_tier:
        sbe = sum(int(r["words"]["spans_before_endpoint"]) for r in rows)  # type: ignore[index]
        print(f"    spans completed before the endpoint: {sbe}")
    if args.two_tier:
        cov = [float(r["words"]["base_covered_frac"]) for r in rows]  # type: ignore[index]
        q = sum(int(r["words"]["spans_queued"]) for r in rows)  # type: ignore[index]
        dn = sum(int(r["words"]["spans_done"]) for r in rows)  # type: ignore[index]
        print(f"\n  TWO-TIER: {dn}/{q} spans re-decoded by base before the endpoint;")
        print(f"  base covered {statistics.median(cov):.0%} of the audio tiny committed")
        print("  (the rest is audio tiny committed and base never reached, which")
        print("  the final still has to decode — it bought nothing).")

    tail = statistics.median([float(r["words"]["tail_s"]) for r in rows])  # type: ignore[index]
    full = statistics.median([float(r["audio_s"]) for r in rows])
    print(f"\n  final decodes {tail:.2f} s of {full:.2f} s = {tail / full:.0%} of the audio")
    print(
        f"  final decode  {statistics.median([float(r['words']['tail_ms']) for r in rows]):.0f} ms "  # type: ignore[index]
        f"against baseline {med('baseline_ms'):.0f} ms"
    )


if __name__ == "__main__":
    main()
