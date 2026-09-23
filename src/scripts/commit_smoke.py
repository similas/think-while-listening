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
import json
import statistics
import time
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

from twl.commit import LocalAgreementCommitter, Word
from twl.pacing import SelfPacedIssuer
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

    tiny = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=3)
    base = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=3)
    for m in (tiny, base):
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
        for want_words in (True, False):
            committer = LocalAgreementCommitter(
                agreement_n=args.agreement_n, tail_guard_s=args.tail_guard_s
            )
            issuer = SelfPacedIssuer(duty_max=args.duty_max)
            now_s, hyp_ms, n_hyp = 0.0, [], 0
            idle_ms = 1e9  # nothing has run yet
            while now_s < dur:
                d = issuer.decide(
                    in_flight=False,
                    uncommitted_s=now_s - committer.committed_end_s,
                    idle_ms=idle_ms,
                    buffer_s=now_s - committer.committed_end_s,
                )
                if not d.issue:
                    now_s += 0.05
                    idle_ms += 50
                    continue
                start = committer.committed_end_s
                buf = pcm[int(start * rate) : int(now_s * rate)]
                if len(buf) < int(0.2 * rate):
                    now_s += 0.05
                    continue
                t1 = time.perf_counter_ns()
                segs = decode(
                    tiny,
                    buf,
                    word_timestamps=want_words,
                    initial_prompt=committer.prompt() or None,
                )
                ms = (time.perf_counter_ns() - t1) / 1e6
                hyp_ms.append(ms)
                n_hyp += 1
                committer.offer(words_of(segs, start, want_words), buffer_end_s=now_s)
                issuer.note_decode(ms)
                # The pacing gap is wall time the utterance also spends.
                now_s += (ms + issuer.required_idle_ms(now_s - committer.committed_end_s)) / 1000.0
                idle_ms = 0.0
            # RACE vs WAIT. The loop above advanced wall time past each decode,
            # so the hypothesis "in flight" at the endpoint is the one whose
            # decode would still have been running when the audio ran out.
            race_from = committer.committed_end_s
            in_flight_ms = max(0.0, (now_s - dur) * 1000.0)
            tail_from = race_from
            t2 = time.perf_counter_ns()
            tail_segs = decode(base, pcm[int(tail_from * rate) :])
            tail_ms = (time.perf_counter_ns() - t2) / 1e6
            tail_text = " ".join(
                s.text.strip()
                for s in tail_segs
                if s.text.strip()  # type: ignore[union-attr]
            ).strip()
            final = f"{committer.committed_text()} {tail_text}".strip()
            per_mode[want_words] = {
                "text": final,
                "committed_end_s": tail_from,
                "tail_s": dur - tail_from,
                "tail_ms": tail_ms,
                "hyp_ms": hyp_ms,
                "n_hyp": n_hyp,
                # What WAIT would have paid: the remaining decode of the
                # hypothesis race discards, before the tail final can start.
                "wait_extra_ms": in_flight_ms,
            }

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

    tail = statistics.median([float(r["words"]["tail_s"]) for r in rows])  # type: ignore[index]
    full = statistics.median([float(r["audio_s"]) for r in rows])
    print(f"\n  final decodes {tail:.2f} s of {full:.2f} s = {tail / full:.0%} of the audio")
    print(
        f"  final decode  {statistics.median([float(r['words']['tail_ms']) for r in rows]):.0f} ms "  # type: ignore[index]
        f"against baseline {med('baseline_ms'):.0f} ms"
    )


if __name__ == "__main__":
    main()
