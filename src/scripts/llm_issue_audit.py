"""Did the LLM request ever precede the endpoint, on a turn that counts?

COMMIT-WL moves recognizer work into the listening window. It must NOT move the
LLM: the reply has to be generated from the final transcript, after the user
stopped. This script checks that on the data, per valid turn, for both arms.

THE REQUEST TIME IS A PROXY, and the proxy is named here rather than buried.
Nothing in the run records the moment the POST leaves the process. The LLM is
started by a TranscriptionFrame, which SttService pushes on the line after it
marks ``stt_final`` (src/twl/stt.py), and LlamaChatProcessor creates the
generate task on receipt. So:

    stt_final  <=  real request time  <  llm_first_token

``stt_final - vad_user_stopped`` is therefore a LOWER BOUND on the request's
offset from the endpoint: if it is non-negative, the request cannot have
preceded the endpoint. The unmeasured residual (frame push + one processor hop)
is strictly positive, so it can only move the request later, never earlier.
A negative value is the only way this audit can report a violation.

Usage:
    python src/scripts/llm_issue_audit.py --arm NOPARTIAL dir [dir ...] \
                                          --arm COMMIT-WL dir [dir ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

STAGES = ("vad_user_stopped", "stt_final", "llm_first_token")


def _ttfa(st: dict) -> float | None:
    """TTFA exactly as score_run.py computes it: audio_out_first - speech_end_est."""
    if "audio_out_first" in st and "speech_end_est" in st:
        return st["audio_out_first"] - st["speech_end_est"]
    return None


def _out_of_order(st: dict) -> bool:
    """The reply began before the endpoint was marked on this turn's clock.

    A turn cannot answer a question the user has not finished asking. When this
    holds, the turn's origin belongs to a different stretch of speech than its
    endpoint mark does - the signature of a VAD split whose second half was not
    flagged - and every offset on that clock, TTFA included, is measured from
    the wrong zero.
    """
    a, v = st.get("audio_out_first"), st.get("vad_user_stopped")
    return a is not None and v is not None and a < v


def read_run(d: Path) -> list[dict]:
    """Per-turn rows: validity, the three stage offsets, reply and transcript."""
    stages: dict[int, dict[str, float]] = {}
    turns: dict[int, dict] = {}
    for line in (d / "turns.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        kind = r.get("kind")
        if kind == "stage_event" and r["stage"] in STAGES:
            # FIRST mark of each stage; repeats are later boundaries, not this one.
            stages.setdefault(r["turn"], {}).setdefault(r["stage"], r["t_ms"])
        elif kind == "turn_record":
            turns[r["turn"]] = r
    rows = []
    for n, rec in sorted(turns.items()):
        s = stages.get(n, {})
        rows.append(
            {
                "run": d.name,
                "turn": n,
                "valid": bool(rec.get("valid")),
                "invalid_reason": rec.get("invalid_reason") or "",
                "endpoint_ms": s.get("vad_user_stopped"),
                "final_ms": s.get("stt_final"),
                "first_token_ms": s.get("llm_first_token"),
                "ttfa_ms": _ttfa(rec.get("stages_ms") or {}),
                "out_of_order": _out_of_order(rec.get("stages_ms") or {}),
                "reply": (rec.get("reply") or "").strip(),
                "transcript": (rec.get("transcript") or "").strip(),
            }
        )
    return rows


def summarize(arm: str, rows: list[dict]) -> list[dict]:
    """Print the two offsets over valid turns and return the violations."""
    valid = [r for r in rows if r["valid"]]
    print(f"\n=== {arm} — {len(valid)} valid turns of {len(rows)} ===")

    def offsets(key: str) -> list[float]:
        return [
            r[key] - r["endpoint_ms"]
            for r in valid
            if r[key] is not None and r["endpoint_ms"] is not None
        ]

    for key, label in (("final_ms", "stt_final"), ("first_token_ms", "llm_first_token")):
        v = offsets(key)
        missing = len(valid) - len(v)
        if not v:
            print(f"  {label:16s} - endpoint:  no data ({missing} turns missing a mark)")
            continue
        v.sort()
        p95 = v[min(len(v) - 1, round(0.95 * (len(v) - 1)))]
        note = f"   [{missing} turns missing a mark]" if missing else ""
        print(
            f"  {label:16s} - endpoint:  median {statistics.median(v):8.1f} ms"
            f"   p95 {p95:8.1f}   min {v[0]:8.1f}   max {v[-1]:8.1f}   n={len(v)}{note}"
        )

    # THE VIOLATION: a request that cannot have followed the endpoint.
    bad = [
        r
        for r in valid
        if r["final_ms"] is not None
        and r["endpoint_ms"] is not None
        and r["final_ms"] < r["endpoint_ms"]
    ]
    print(f"  requests preceding the endpoint on a VALID turn: {len(bad)}")
    for r in bad:
        print(f"    {r['run']} turn {r['turn']}: {r['final_ms'] - r['endpoint_ms']:+.1f} ms")
        print(f"      transcript: {r['transcript']!r}")
        print(f"      reply:      {r['reply']!r}")

    # THE CONTAMINATION, and what it does to the headline number.
    oo = [r for r in valid if r["out_of_order"]]
    print(f"  valid turns whose reply PRECEDES their endpoint mark: {len(oo)}")
    for r in oo:
        t = r["ttfa_ms"]
        print(
            f"    {r['run']} turn {r['turn']}: TTFA {t:.1f} ms"
            if t is not None
            else f"    {r['run']} turn {r['turn']}: no TTFA"
        )
    kept = [r["ttfa_ms"] for r in valid if r["ttfa_ms"] is not None and not r["out_of_order"]]
    allt = [r["ttfa_ms"] for r in valid if r["ttfa_ms"] is not None]
    if allt:
        print(
            f"  TTFA median  as scored {statistics.median(allt):8.1f} ms (n={len(allt)})"
            f"   flagged out {statistics.median(kept):8.1f} ms (n={len(kept)})"
            f"   shift {statistics.median(kept) - statistics.median(allt):+.1f} ms"
        )

    # Reported separately: turns that count but lost a mark, which is a hole in
    # the stage decomposition even though TTFA (tts_first/audio_out) is intact.
    holes = [r for r in valid if r["first_token_ms"] is None]
    if holes:
        print(f"  valid turns MISSING llm_first_token: {len(holes)}")
        for r in holes:
            print(f"    {r['run']} turn {r['turn']}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        action="append",
        nargs="+",
        metavar=("NAME", "DIR"),
        required=True,
        help="arm name followed by one or more run directories",
    )
    args = ap.parse_args()
    total_bad = 0
    for spec in args.arm:
        arm, dirs = spec[0], [Path(p) for p in spec[1:]]
        rows: list[dict] = []
        for d in dirs:
            rows.extend(read_run(d))
        total_bad += len(summarize(arm, rows))
    print(f"\nTOTAL requests preceding the endpoint on valid turns: {total_bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
