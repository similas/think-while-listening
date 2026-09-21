"""Did any GPU-requesting run silently execute on the CPU? Segment-level audit.

The CPU-fallback guard in llama_server.sh was dead from 8afb9b1 (2026-09-16
10:30) until 2026-09-21: LOG_OFFSET was read but never assigned, the expansion
failed inside a subshell under ``set -u``, and the guard reported a clean start
without ever inspecting the log. Every run started in that window is unverified
by the guard and has to be verified after the fact, from evidence.

THE RECORDED COMMAND LINE IS NOT EVIDENCE. run_meta.software carries
"--n-gpu-layers 99", but that is what was REQUESTED. A server whose CUDA backend
fails to load is started with exactly those flags and ignores them. Only the
server's own output says what happened, so this script reads nothing else.

Three facts per server process, all from results/raw/llama-server.log:

  1. whether "no usable GPU found" was printed,
  2. the generation speed the server reported (print_timing "eval time"),
  3. when the process ran, so runs can be attributed to it.

(1) and (2) are in the log; (3) is not, because llama.cpp timestamps are
process-relative (mm.ss.mmm.uuu from its own start). Absolute time comes from
the journal's "Started" records, aligned to the log's segments below.

THE WARNING BELONGS TO THE SEGMENT THAT FOLLOWS IT. It is printed during
argument parsing, before the logger is initialised, so it lands in the file with
no timestamp, after the PREVIOUS process's "cleaning up before exit" and before
this process's first timestamped line. Attributing it to the preceding segment
(the naive reading) blames a process that decoded at full GPU speed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
import statistics
import subprocess
from pathlib import Path

WARNING = "no usable GPU found"
SEGMENT_START = "common_params_print_info"
# mm.ss.mmm.uuu, counted from this process's own start.
TS = re.compile(r"^(\d+)\.(\d\d)\.(\d\d\d)\.(\d\d\d) ")
# The GENERATION timing. "prompt eval time" is a different line and a different
# measurement; the leading "|" keeps it out.
EVAL = re.compile(r"\|\s+eval time =\s+[\d.]+ ms /\s+(\d+) tokens \(\s+([\d.]+) ms per token")
STARTED = re.compile(r"(20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d{4}).*Started.*--n-gpu-layers (\d+)")
# Generations shorter than this are warm-ups and single-token stops; their
# per-token figure is dominated by fixed cost.
MIN_TOKENS = 8
GUARD_BROKE = dt.datetime.fromisoformat("2026-09-16T10:30:13-04:00")
STARTUP_SLACK_S = 60.0
# StubLlmProcessor.REPLY. A run whose every reply is this sentence never
# called the server, so its segment has no generations to time — and its
# run_meta notes do not always say so.
STUB_REPLY = "Yes, I can hear you clearly."


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def parse_segments(log: Path) -> list[dict]:
    """One entry per server process, in file order."""
    segs: list[dict] = []
    pending_warning = False
    for line in log.read_text(errors="replace").splitlines():
        if WARNING in line:
            pending_warning = True
        if SEGMENT_START in line:
            segs.append({"warning": pending_warning, "dur_s": 0.0, "rates": []})
            pending_warning = False
        if not segs:
            continue
        m = TS.match(line)
        if m:
            mins, secs, ms, _us = (int(x) for x in m.groups())
            segs[-1]["dur_s"] = mins * 60 + secs + ms / 1e3
        e = EVAL.search(line)
        if e and int(e.group(1)) >= MIN_TOKENS:
            segs[-1]["rates"].append(float(e.group(2)))
    return segs


def journal_starts(unit: str, cache: Path) -> list[tuple[dt.datetime, int]]:
    """Absolute start times, from the journal, cached to the repo.

    THE JOURNAL IS VOLATILE. It rotates on size and age, so the evidence that
    dates these segments will disappear from this machine and cannot be
    regenerated. Every reading is therefore merged into ``cache`` and committed
    with the results, which is what makes this audit reproducible later and on
    another machine. The cache lives under results/ rather than results/raw/
    because it is derived provenance — start times and a flag value, no log
    text and no audio — and results/raw/ stays local by CLAUDE.md §1.
    """
    starts: dict[str, int] = {}
    if cache.exists():
        starts.update(json.loads(cache.read_text()))
    out = subprocess.run(
        ["journalctl", "--user", "-u", unit, "--no-pager", "-o", "short-iso"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    for line in out.splitlines():
        m = STARTED.search(line)
        if m:
            starts[m.group(1)] = int(m.group(2))
    if not starts:
        raise SystemExit(f"no starts in the journal and none cached in {cache}")
    cache.write_text(json.dumps(dict(sorted(starts.items())), indent=1) + "\n")
    return sorted((dt.datetime.fromisoformat(f"{k[:-2]}:{k[-2:]}"), v) for k, v in starts.items())


def align(segs: list[dict], starts: list[tuple[dt.datetime, int]]) -> int:
    """Offset k such that log segment i is the process started at starts[i + k].

    The log holds fewer segments than the journal holds starts, because the log
    file postdates the first start. The alignment is pinned by one physical
    constraint: a process cannot outlive the start of its successor, so
    dur[i] <= starts[i+k+1] - starts[i+k] for every i. On this data exactly one
    offset survives, and several segments match their gap to within seconds.
    """
    feasible = []
    for k in range(len(starts) - len(segs) + 1):
        if all(
            segs[i]["dur_s"] <= (starts[i + k + 1][0] - starts[i + k][0]).total_seconds() + 1
            for i in range(len(segs) - 1)
            if i + k + 1 < len(starts)
        ):
            feasible.append(k)
    if len(feasible) != 1:
        raise SystemExit(f"alignment is not unique: offsets {feasible} all fit")
    return feasible[0]


def load_runs() -> list[dict]:
    """(wall time, run id, notes, requested ngl) for every run on disk."""
    runs = []
    for path in sorted(glob.glob("results/raw/*/*/turns.jsonl")):
        try:
            with open(path, encoding="utf-8") as fh:
                meta = json.loads(fh.readline())
        except Exception:
            continue
        if meta.get("kind") != "run_meta" or "wall_time" not in meta:
            continue
        stamp = meta["wall_time"]
        cmdline = (meta.get("software") or {}).get("llama-server-cmdline", "")
        ngl = re.search(r"--n-gpu-layers (\d+)", cmdline)
        replies = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("kind") == "turn_record":
                    replies.append((row.get("reply") or "").strip())
        runs.append(
            {
                "at": dt.datetime.fromisoformat(f"{stamp[:-2]}:{stamp[-2:]}"),
                "id": meta["run_id"],
                "notes": meta.get("notes", ""),
                "ngl": int(ngl.group(1)) if ngl else None,
                "stub": bool(replies) and all(r == STUB_REPLY for r in replies),
            }
        )
    return sorted(runs, key=lambda r: r["at"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, default=Path("results/raw/llama-server.log"))
    p.add_argument("--unit", default="twl-llama.service")
    p.add_argument("--starts", type=Path, default=Path("results/llama_server_starts.json"))
    p.add_argument("--runs", action="store_true", help="list every post-break run")
    args = p.parse_args()

    segs = parse_segments(args.log)
    starts = journal_starts(args.unit, args.starts)
    k = align(segs, starts)
    for i, s in enumerate(segs):
        s["at"], s["ngl"] = starts[i + k]
        # A process is live from its start until it logs "cleaning up before
        # exit" — the last timestamped line of its segment. Ending it at the
        # NEXT start instead would sweep the gaps in between into the
        # preceding server, and those gaps are an experimental condition here:
        # the "llama stopped" attribution arm runs with no server at all.
        s["until"] = s["at"] + dt.timedelta(seconds=s["dur_s"] + 2)

    print(f"log segments (server processes): {len(segs)}")
    print(
        f"journal starts: {len(starts)}; alignment offset {k} — the first "
        f"{k} start(s) predate the log file"
    )
    print(
        f"\n{'seg':>3} {'started':>20} {'ngl':>4} {'dur_s':>8} {'warned':>7} "
        f"{'decodes':>8} {'ms/token':>9}"
    )
    for i, s in enumerate(segs):
        print(
            f"{i:>3} {s['at'].strftime('%Y-%m-%d %H:%M:%S'):>20} {s['ngl']:>4} "
            f"{s['dur_s']:>8.0f} {'WARNED' if s['warning'] else '-':>7} "
            f"{len(s['rates']):>8} {median(s['rates']):>9.1f}"
        )

    warned = [s for s in segs if s["warning"]]
    offload = [s for s in segs if not s["warning"] and s["ngl"] > 0]
    no_offload = [s for s in segs if not s["warning"] and s["ngl"] == 0]
    print(
        f"\nsegments that warned: {len(warned)} — "
        f"{sum(len(s['rates']) for s in warned)} generations between them"
    )
    print(f"  requested ngl on those: {sorted({s['ngl'] for s in warned})}")
    print(
        f"segments that did NOT warn and requested ngl>0: {len(offload)}, "
        f"median {median([r for s in offload for r in s['rates']]):.1f} ms/token "
        f"(n={sum(len(s['rates']) for s in offload)} generations)"
    )
    cpu = [r for s in no_offload for r in s["rates"]]
    print(
        f"segments that did NOT warn and requested ngl=0: {len(no_offload)}, "
        f"median {median(cpu):.1f} ms/token (n={len(cpu)} generations)"
    )

    runs = load_runs()
    post = [r for r in runs if r["at"] > GUARD_BROKE]
    print(f"\nruns on disk: {len(runs)}; started after the guard broke: {len(post)}")
    unattributed, in_warned = [], []
    for r in post:
        seg = next((i for i, s in enumerate(segs) if s["at"] <= r["at"] < s["until"]), None)
        if seg is None:
            # run_meta.wall_time is stamped when the RUN starts, and the
            # harness starts the server immediately after, so a run can
            # legitimately predate its own server by a few seconds. A gap of
            # minutes is a different thing entirely — it is the "llama
            # stopped" arm — so only a start that follows within STARTUP_SLACK
            # adopts the run.
            seg = next(
                (
                    i
                    for i, s in enumerate(segs)
                    if 0 <= (s["at"] - r["at"]).total_seconds() <= STARTUP_SLACK_S
                ),
                None,
            )
        r["seg"] = seg
        if seg is None:
            unattributed.append(r)
        elif segs[seg]["warning"]:
            in_warned.append(r)
    print(f"  ran while a server was live: {len(post) - len(unattributed)}")
    print(f"  ran with NO server live: {len(unattributed)}")
    for r in unattributed:
        print(f"    {r['id']}  {r['at']:%Y-%m-%d %H:%M:%S}  {r['notes'].split(';')[-1].strip()}")
    print(f"  in a segment that warned: {len(in_warned)}")
    for r in in_warned:
        print(f"    {r['id']}  seg {r['seg']}  {r['notes'].split(';')[-1].strip()}")
    print("  A warning is only a fault if offload was WANTED. Compare each one")
    print("  against the arm it was run for, above: a control arm that asked for")
    print("  no CUDA backend and got none is doing what it was told.")

    # Every post-break run, through its segment. A run in a segment with no
    # generations has no decode rate because the server never decoded during
    # it — the stub-LLM attribution arms hold a resident server and never call
    # it — not because the rate is missing.
    rated = [(r, median(segs[r["seg"]]["rates"])) for r in post if r["seg"] is not None]
    have = [v for _, v in rated if v == v]
    none = [r for r, v in rated if v != v]
    print(f"\n{len(rated)} post-break runs carry their segment's decode rate:")
    print(
        f"  with a rate: {len(have)} runs, median {median(have):.1f} ms/token, "
        f"range {min(have):.1f}-{max(have):.1f}"
    )
    print(f"  no rate: {len(none)} runs — their server decoded nothing while they ran")
    print(f"    of which every reply is the stub sentence: {sum(1 for r in none if r['stub'])}")
    live = [r for r, v in rated if v == v and not r["stub"]]
    live_rates = [median(segs[r["seg"]]["rates"]) for r in live]
    print(f"\nruns that actually called the server: {len(live)}")
    print(f"  their segments' decode rates: {min(live_rates):.1f}-{max(live_rates):.1f} ms/token")
    print(
        f"  of them in a segment that warned: {sum(1 for r in live if segs[r['seg']]['warning'])}"
    )

    if args.runs:
        print(f"\n{'run':>38} {'seg':>4} {'warned':>7} {'ms/token':>9}  notes")
        for r in post:
            s = segs[r["seg"]] if r["seg"] is not None else None
            print(
                f"{r['id']:>38} {r['seg']!s:>4} "
                f"{('WARNED' if s and s['warning'] else '-'):>7} "
                f"{(median(s['rates']) if s else float('nan')):>9.1f}  {r['notes'][:60]}"
            )


if __name__ == "__main__":
    main()
