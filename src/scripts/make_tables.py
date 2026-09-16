"""Build results/tables/*.md from results/raw/ logs — the `make results` entry.

Every quantitative claim in NOTES, reports, and (later) the paper traces to a
table produced here from a raw JSONL log. Deterministic: same logs, same
bytes out. Runs that are absent are skipped with a note, never invented.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from twl.metrics import median, summarize
from twl.records import read_jsonl

REPO = Path(__file__).resolve().parents[2]

STAGE_ORDER = (
    "vad_user_stopped",
    "stt_final",
    "llm_first_token",
    "llm_done",
    "tts_first_audio",
    "audio_out_first",
    "playback_done",
)


def fmt(x: float) -> str:
    return f"{x:.0f}"


def table_prefix_reuse(raw: Path, out: list[str]) -> None:
    logs = sorted((raw / "prefix_reuse").glob("*.jsonl"))
    if not logs:
        out.append("## KV-prefix reuse\n\n(no runs)\n")
        return
    log = logs[-1]
    rows = [r for r in read_jsonl(str(log)) if r.get("kind") == "prefix_reuse_sample"]
    meta = next(r for r in read_jsonl(str(log)) if r.get("kind") == "run_meta")
    out.append(f"## KV-prefix reuse — {meta['run_id']}\n")
    out.append(f"git {meta['git_commit'][:12]}, {meta['nvpmodel']}\n")
    out.append("| growth | cache_prompt | n | median prompt_n | median prompt_ms |")
    out.append("|---|---|---|---|---|")
    for tail_free in (True, False):
        for cached in (True, False):
            sel = [
                r
                for r in rows
                if r["tail_free"] is tail_free
                and r["cache_prompt"] is cached
                and r["step"] > 0
                and not r["final"]
            ]
            if sel:
                pn = median([float(r["prompt_n"]) for r in sel])
                pm = median([float(r["prompt_ms"]) for r in sel])
                growth = "tail-free" if tail_free else "templated"
                out.append(f"| {growth} | {cached} | {len(sel)} | {fmt(pn)} | {pm:.1f} |")
    cold = [float(r["prompt_ms"]) for r in rows if r["step"] == 0]
    out.append(f"\ncold step-0 prefills: n={len(cold)}, median {median(cold):.1f} ms\n")


def reactive_run_rows(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    recs = read_jsonl(str(run_dir / "turns.jsonl"))
    meta = next(r for r in recs if r.get("kind") == "run_meta")
    turns = [r for r in recs if r.get("kind") == "turn_record"]
    return meta, turns


def invalidation_summary(turns: list[dict[str, Any]]) -> str:
    """Per-run invalidation counts by reason — flagged rows are kept, never dropped."""
    reasons: dict[str, int] = {}
    for t in turns:
        if not t["valid"]:
            key = t["invalid_reason"].split(":")[0] or "unspecified"
            reasons[key] = reasons.get(key, 0) + 1
    if not reasons:
        return "0"
    return (
        f"{sum(reasons.values())} ("
        + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items()))
        + ")"
    )


def table_reactive(raw: Path, out: list[str]) -> None:
    runs = sorted((raw / "reactive").glob("reactive-*"))
    if not runs:
        out.append("## REACTIVE runs\n\n(no runs)\n")
        return
    out.append("## REACTIVE runs\n")
    out.append(
        "| run | turns | invalid | "
        + " | ".join(s.replace("_", " ") + " med ms" for s in STAGE_ORDER[:6])
        + " | TTFA med ms | TTFA p95 ms |"
    )
    out.append("|---|---|---|" + "---|" * 8)
    for run_dir in runs:
        if not (run_dir / "turns.jsonl").exists():
            continue
        _meta, turns = reactive_run_rows(run_dir)
        valid = [t for t in turns if t["valid"]]
        cells = [run_dir.name, str(len(turns)), invalidation_summary(turns)]
        for stage in STAGE_ORDER[:6]:
            xs = [t["stages_ms"][stage] for t in valid if stage in t["stages_ms"]]
            cells.append(fmt(median(xs)) if xs else "—")
        ttfa = [
            t["stages_ms"]["audio_out_first"] - t["stages_ms"]["speech_end_est"]
            for t in valid
            if "audio_out_first" in t["stages_ms"] and "speech_end_est" in t["stages_ms"]
        ]
        if ttfa:
            s = summarize(ttfa, n_resamples=2000)
            cells.append(f"{s.median:.0f} [{s.median_ci[0]:.0f},{s.median_ci[1]:.0f}]")
            cells.append(fmt(s.p95))
        else:
            cells += ["—", "—"]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")


def table_validation(raw: Path, out: list[str]) -> None:
    comps = sorted((raw / "validate_playback").glob("*/comparison.json"))
    if not comps:
        out.append("## Playback-harness validation\n\n(no completed comparison)\n")
        return
    data = json.loads(comps[-1].read_text())
    out.append(f"## Playback-harness validation — {comps[-1].parent.name}\n")
    out.append(
        f"turns: file={data['n_file']} file2={data['n_file2']} mic={data['n_mic']}; "
        f"transcript mismatches (mic vs file): {len(data['transcript_mismatches'])}\n"
    )
    out.append(
        "| stage | mic-file median | mic-file worst | floor (file2-file) worst | within floor |"
    )
    out.append("|---|---|---|---|---|")
    for stage, d in data["stages"].items():
        out.append(
            f"| {stage} | {d['mic_median_ms']:+.1f} | {d['mic_worst_abs_ms']:.1f} | "
            f"{d['floor_worst_abs_ms']:.1f} | {d['within_noise_floor']} |"
        )
    out.append("")


def table_stt_isolation(raw: Path, out: list[str]) -> None:
    logs = sorted((raw / "stt_isolation").glob("*.jsonl"))
    if not logs:
        out.append("## STT isolation\n\n(no runs)\n")
        return
    rows = [r for r in read_jsonl(str(logs[-1])) if r.get("kind") == "stt_isolation_sample"]
    xs = [float(r["decode_ms"]) for r in rows]
    s = summarize(xs, n_resamples=2000)
    out.append(f"## STT isolation — {logs[-1].stem}\n")
    out.append(
        f"n={s.n}, median {s.median:.0f} ms [{s.median_ci[0]:.0f}, {s.median_ci[1]:.0f}], "
        f"p95 {s.p95:.0f} ms\n"
    )


def table_memory(raw: Path, out: list[str]) -> None:
    runs = sorted((raw / "reactive").glob("reactive-*"))
    best: tuple[int, Path] | None = None
    for run_dir in runs:  # the longest run is the memory-creep evidence
        if (run_dir / "turns.jsonl").exists():
            _, turns = reactive_run_rows(run_dir)
            if best is None or len(turns) > best[0]:
                best = (len(turns), run_dir)
    if best is None or best[0] < 20:
        out.append("## Memory over turns\n\n(no run long enough)\n")
        return
    _, turns = reactive_run_rows(best[1])
    out.append(f"## Memory over turns — {best[1].name} ({best[0]} turns)\n")
    out.append("| metric | turn 1 | mid | last | max |")
    out.append("|---|---|---|---|---|")
    for proc in sorted(turns[0].get("rss_mb", {})):
        xs = [t["rss_mb"].get(proc, -1.0) for t in turns]
        out.append(
            f"| {proc} RSS MB | {xs[0]:.0f} | {xs[len(xs) // 2]:.0f} | {xs[-1]:.0f} "
            f"| {max(xs):.0f} |"
        )
    ma = [t["mem_available_mb"] for t in turns]
    mid_ma = ma[len(ma) // 2]
    out.append(
        f"| MemAvailable MB | {ma[0]:.0f} | {mid_ma:.0f} | {ma[-1]:.0f} | {min(ma):.0f} (min) |"
    )
    sw = [sum(t["swap_used_mb"].values()) for t in turns]
    out.append(
        f"| swap total MB | {sw[0]:.1f} | {sw[len(sw) // 2]:.1f} | {sw[-1]:.1f} | {max(sw):.1f} |"
    )
    out.append(f"\ninvalid turns: {sum(1 for t in turns if not t['valid'])} / {len(turns)}\n")


def table_capture_hazards(raw: Path, out_dir: Path) -> None:
    """Appendix table: what the capture path can silently cost you.

    Kept separate from the phase tables because it is a hazards appendix, not
    a result: it documents that opening a multi-channel array with one channel
    downmixes it, and what that costs in word error rate.
    """
    comps = sorted((raw / "capture_channels").glob("channel-compare-*.json"))
    ids = sorted((raw / "capture_channels").glob("capture-channels-*.json"))
    lines = ["# Capture-path hazards (generated by src/scripts/make_tables.py)\n"]
    if not comps:
        lines.append("(no channel comparison recorded)\n")
        (out_dir / "capture_hazards.md").write_text("\n".join(lines))
        return

    data = json.loads(comps[-1].read_text())
    cap = data.get("capture", {})
    product = cap.get("usb_product", "?")
    firmware = cap.get("usb_firmware_bcd_device", "?")
    lines.append(f"Device: {product}, firmware {firmware}, {cap.get('mode', '?')}.\n")
    lines.append(
        f"Run `{data['run_id']}`, git `{data['git_commit'][:12]}`, "
        f"{len(data['per_utterance'])} known utterances played through the room speaker "
        f"and decoded with the pipeline's own STT model.\n"
    )
    lines.append("| input to the recognizer | mean WER |")
    lines.append("|---|---|")
    label = {
        "ch0": "channel 0 alone",
        "ch1": "channel 1 alone (chosen)",
        "mean": "downmix of both (what a 1-channel open produces)",
    }
    for cand, value in data["mean_wer"].items():
        lines.append(f"| {label.get(cand, cand)} | {value:.3f} |")
    worst = max(
        (u for u in data["per_utterance"]),
        key=lambda u: float(u["mean"]["wer"]),
    )
    lines.append(
        f"\nWorst downmix failure: reference {worst['reference']!r} decoded as "
        f"{worst['mean']['text']!r} (WER {float(worst['mean']['wer']):.2f}), while channel 1 "
        f"gave {worst['ch1']['text']!r}.\n"
    )

    if ids:
        idd = json.loads(ids[-1].read_text())
        lines.append("## Is the array's echo canceller being fed?\n")
        lines.append(
            f"Log sweep, normalized cross-correlation with lag search; correlation floor "
            f"(no playback) ncc {idd['correlation_floor_ncc']:.3f}.\n"
        )
        lines.append("| condition | channel | peak ncc | lag ms | tail ratio | rms |")
        lines.append("|---|---|---|---|---|---|")
        for cond, chans in idd["conditions"].items():
            for ch, m in chans.items():
                lines.append(
                    f"| {cond} | {ch} | {m['peak_ncc']:.3f} | {m['peak_lag_ms']:.2f} | "
                    f"{m['tail_ratio']:.3f} | {m['rms']:.5f} |"
                )
        for v in idd["verdicts"]:
            lines.append(f"\n- {v}")
        lines.append(
            "\n- The array-output condition transduced no sound (nothing is connected to the "
            "3.5 mm jack), so its null result is no evidence either way; see results/NOTES.md."
        )
    (out_dir / "capture_hazards.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=REPO / "results/raw")
    p.add_argument("--out-dir", type=Path, default=REPO / "results/tables")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out: list[str] = ["# Phase 1 tables (generated by src/scripts/make_tables.py)\n"]
    table_prefix_reuse(args.raw_dir, out)
    table_reactive(args.raw_dir, out)
    table_validation(args.raw_dir, out)
    table_stt_isolation(args.raw_dir, out)
    table_memory(args.raw_dir, out)
    (args.out_dir / "phase1.md").write_text("\n".join(out) + "\n")
    print(f"wrote {args.out_dir / 'phase1.md'}")
    table_capture_hazards(args.raw_dir, args.out_dir)
    print(f"wrote {args.out_dir / 'capture_hazards.md'}")

    # CSV twin of the reactive table for downstream plotting.
    runs = sorted((args.raw_dir / "reactive").glob("reactive-*"))
    with open(args.out_dir / "reactive_runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "turn", "valid", *STAGE_ORDER, "ttfa_ms", "transcript_chars"])
        for run_dir in runs:
            if not (run_dir / "turns.jsonl").exists():
                continue
            _, turns = reactive_run_rows(run_dir)
            for t in turns:
                st = t["stages_ms"]
                ttfa = (
                    st["audio_out_first"] - st["speech_end_est"]
                    if "audio_out_first" in st and "speech_end_est" in st
                    else ""
                )
                w.writerow(
                    [
                        run_dir.name,
                        t["turn"],
                        t["valid"],
                        *[round(st.get(s, -1), 1) for s in STAGE_ORDER],
                        ttfa if ttfa == "" else round(ttfa, 1),
                        len(t["transcript"]),
                    ]
                )
    print(f"wrote {args.out_dir / 'reactive_runs.csv'}")


if __name__ == "__main__":
    main()
