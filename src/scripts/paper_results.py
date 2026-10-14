"""Tables 1-3 and results/RESULTS.md, from the claim registry, the ledger and the
pass 6 decision log. `make results` runs it.

Table 1  where to speculate: listener vs thinker, every cell a scorer line.
Table 2  the allocator's decisions by reason (pass 6), from the decision records.
Table 3  the pre-registration ledger: each prediction, its registration commit,
         its scoring commit, its verdict as it stands, and correction commits.
RESULTS  one page: every claim the paper makes, quoted from its scorer, with the
         runs, the command and the NOTES entry. The paper is written from it.

Commits are resolved from git (the first commit whose NOTES diff adds the
heading), never typed. The ledger's verdict lines are curated from NOTES
(src/scripts/paper_ledger.json); every one names the heading it summarises, so
each can be checked against the entry.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import paper_sources as ps

REPO = ps.REPO
TABLES = REPO / "results/tables"
LEDGER = Path(__file__).with_name("paper_ledger.json")
NOTES = "results/NOTES.md"


@cache
def commit_of(heading: str) -> str:
    out = subprocess.run(
        ["git", "log", "-S", heading, "--reverse", "--format=%h %ad", "--date=short", "--", NOTES],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if not out:
        raise LookupError(f"no commit adds the NOTES heading {heading!r}")
    return out[0]


def claim(key: str) -> ps.Claim:
    hits = [c for c in ps.CLAIMS if c.key == key]
    if len(hits) != 1:
        raise LookupError(key)
    return hits[0]


def grab(key: str, pattern: str) -> str:
    """Group 1 of pattern in claim key's quoted line."""
    m = re.search(pattern, claim(key).quote())
    if not m:
        raise LookupError(f"{key}: {pattern!r} not in {claim(key).quote()!r}")
    return m.group(1)


CI = r"([+-]?[0-9.]+ (?:ms|J|J/ms|ms/s)? ?\[[^\]]+\])"


# ------------------------------------------------------------------ Table 1 --
def table1_rows() -> list[tuple[str, str, str]]:
    return [
        (
            "Survival: how much of the speculation is kept",
            "words committed before the endpoint: "
            + grab("p8_listener", r"median (\S+ \[[^\]]+\] n=\d+)")
            + " items (STOP + gate, pass 6; P8)",
            grab("spec_survival", r"per turn median (\d+)")
            + " draft tokens per turn; per-item median of accepted / produced "
            + grab("spec_survival", r"accepted/produced median (\S+ \(n=\d+ items\))")
            + " (pass 7)",
        ),
        (
            "Value: TTFA, paired per item, multi_step",
            "reduction vs REACTIVE-NOPARTIAL "
            + grab("p3", r"(\+\d+ ms \[[^\]]+\] n=\d+)")
            + " (pass 2k)",
            "difference SPEC-CONTINUE - REACTIVE "
            + formatted("SpecTTFAr")
            + " n="
            + grab("spec_ttfa", r"n=(\d+)")
            + ", positive = slower: no saving detected (pass 7, descriptive)",
        ),
        (
            "Overlap cost: a decode in flight at the endpoint",
            "final decode +"
            + grab("listener_c", r"c (\d+ ms/s \[[^\]]+\])")
            + " per s of overlap; "
            + grab("p2a_pop", r"(\d+ turns, \d+ runs)")
            + ", CI over runs; arms with partial decodes in flight (P2a, not the STOP arm, "
            "which freezes them). Partials in flight cost canonical REACTIVE "
            + grab("listener_intervention", r"(\+[0-9.]+ ms \[[^\]]+\] n=\d+)")
            + " of TTFA on dev against NOPARTIAL (pass 1n)",
            "TTFA +"
            + grab("thinker_overlap", r"endpoint: \+([0-9.]+ ms \[[^\]]+\] n=\d+)")
            + " when the draft is still decoding; "
            + grab("thinker_finished", r"endpoint: (\+[0-9.]+ ms \[[^\]]+\] n=\d+)")
            + " when it finished (blocked policy runs, 2026-09-18/20; the onset test found "
            "a finished-draft penalty at one contended cell)",
        ),
        (
            "Energy per turn, net of the board floor",
            "vs REACTIVE-NOPARTIAL "
            + grab("energy_listener", r"j_net: (\+[0-9.]+ J \[[^\]]+\] n=\d+)")
            + " (pass 2k); "
            + grab("p7_jms", r"medians (\+[0-9.]+ J/ms \[[^\]]+\])")
            + " per ms saved; resumed-and-discarded stale decodes "
            + grab("stale_net", r"(mean [0-9.]+ J, total [0-9.]+ J)")
            + " net over "
            + grab("stale", r"(\d+) measured")
            + " decodes",
            "vs REACTIVE "
            + formatted("SpecEnergyr")
            + " n="
            + grab("spec_energy", r"n=(\d+)")
            + " (pass 7, descriptive)",
        ),
        (
            "Memory",
            "the decoding worker process: "
            + grab("worker_memory", r"(PSS median [0-9.]+ MB)")
            + ", MemAvailable "
            + grab("worker_memory", r"pool spawn median (-?[0-9.]+ MB)")
            + " lower (pass 2k, at spawn). Keeping an idle second engine resident: "
            + grab("residency_memory", r"medians\): (\+[0-9.]+ MB)")
            + " RSS, "
            + grab("residency_memory", r"(n=\d+/\d+ runs)")
            + ", no detectable TTFA cost "
            + grab("residency", r"(\+[0-9.]+ ms \[[^\]]+\] n=\d+)")
            + " (pass 1r)",
            "no second model: drafts run in the reply LLM's single slot (--parallel 1, "
            "fixed context); not measured separately",
        ),
    ]


def table1() -> str:
    rows = table1_rows()
    out = [
        "# Table 1. Where to speculate: the listener against the thinker",
        "",
        "Generated by src/scripts/paper_results.py. Each cell quotes a scorer line;",
        "RESULTS.md names the scorer, its runs and its NOTES entry.",
        "",
        "| | Listener (COMMIT-WL-STOP + feasibility gate) | Thinker (SPEC-CONTINUE) |",
        "|---|---|---|",
    ]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    out += [
        "",
        "Sources: survival P8 (fstar_score.py) and pass 7 (p7_score.py); value pass 2k",
        "(p2k_score.py) and pass 7; overlap cost P2a (p2_score.py) and overlap_penalties.py;",
        "energy pass 2k and pass 7; memory and residency pass 1r (p1r_score.py).",
        "Pass 7 is descriptive, its scorer post hoc and its interleave design compromised",
        "(NOTES 2026-10-13).",
    ]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ Table 2 --
def table2() -> str:
    (run,) = ps.PASS6
    recs = [
        json.loads(x)
        for x in (REPO / run / "turns.jsonl").read_text().splitlines()
        if '"decision_record"' in x
    ]
    by: dict[str, list[dict]] = defaultdict(list)
    issued: Counter = Counter()
    for r in recs:
        reason = r["decision"]["inputs"]["reason"]
        if r["decision"]["reason"] != f"allocator: {reason}":
            raise ValueError(f"reason mismatch in turn {r['turn']}")
        by[reason].append(r)
        issued[(reason, r.get("outcome"))] += 1

    def rng(v: list[float], fmt: str) -> str:
        return (
            f"{format(statistics.median(v), fmt)} ({format(min(v), fmt)} to {format(max(v), fmt)})"
        )

    out = [
        "# Table 2. The allocator's decisions by reason (pass 6)",
        "",
        f"Run {Path(run).name}, multi_step, COMMIT-WL-STOP + gate, allocator policy.",
        f"{len(recs)} decision records over {len({r['turn'] for r in recs})} turns. Cells: median",
        "(min to max). Generated by src/scripts/paper_results.py from the decision records.",
        "",
        "| reason | decisions | turns | f_hat | p_usable_hat | feasible budget (tokens) "
        "| remaining (ms) | budget spent | drafts issued |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for reason in sorted(by, key=lambda k: -len(by[k])):
        rs = by[reason]
        inp = [r["decision"]["inputs"] for r in rs]
        spent = sum(1 for r in rs if r["decision"]["budget_tokens"] > 0)
        drafts = sum(n for (rr, o), n in issued.items() if rr == reason and o != "not issued")
        out.append(
            f"| {reason} | {len(rs)} | {len({r['turn'] for r in rs})} "
            f"| {rng([i['f_hat'] for i in inp], '.3f')} "
            f"| {rng([i['p_usable_hat'] for i in inp], '.3f')} "
            f"| {rng([i['feasible_budget'] for i in inp], '.0f')} "
            f"| {rng([i['remaining_ms'] for i in inp], '.0f')} | {spent} | {drafts} |"
        )
    be = {r["decision"]["inputs"]["break_even"] for r in recs}
    rate = {r["decision"]["inputs"]["ms_per_token"] for r in recs}
    out += [
        "",
        f"Break-even p_usable {sorted(be)}; decode rate {sorted(rate)} ms/token on every",
        "decision. Every record carries its inputs and every reason matches them",
        "(NOTES 2026-10-09, PASS 8 (f*), PASS 6 (ALLOCATOR), P8).",
    ]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ Table 3 --
def table3() -> str:
    ledger = json.loads(LEDGER.read_text())
    out = [
        "# Table 3. The pre-registration ledger",
        "",
        "Every registered prediction: the commit that registered its bar, the commit",
        "that scored it, its verdict as it stands after every correction, and the",
        "commits of the corrections. Commits are resolved from git by the NOTES",
        "heading each entry carries. Generated by src/scripts/paper_results.py from",
        "src/scripts/paper_ledger.json.",
        "",
        "| # | prediction | registered | scored | verdict as it stands | corrections |",
        "|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(ledger, 1):
        reg = commit_of(e["registered"])
        if e.get("registered_original"):
            reg = f"{commit_of(e['registered_original'])}; as amended {reg}"
        sc = commit_of(e["scored"]) if e.get("scored") else "not scored"
        corr = ", ".join(commit_of(h).split()[0] for h in e.get("corrections", [])) or "none"
        pred = e["prediction"].replace("|", "/")
        verdict = e["verdict"].replace("|", "/")
        if e.get("notes"):
            verdict += " *Note: " + e["notes"].replace("|", "/") + "*"
        out.append(f"| {i} | **{e['id']}**: {pred} | {reg} | {sc} | {verdict} | {corr} |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ RESULTS --
SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Where the silence goes (Fig 1)",
        (
            "stages_dev",
            "stages_short_digit",
            "stages_long_digit",
            "stages_multi_step",
            "stages_longest",
        ),
    ),
    (
        "The occupancy tax (Fig 2)",
        (
            "fig2_alone",
            "fig2_ballast",
            "tax_ballast_7dc19e",
            "tax_ballast_ace562",
            "tax_full_37443c",
            "tax_full_74e10a",
        ),
    ),
    (
        "Two overlap penalties (Fig 3)",
        (
            "thinker_overlap",
            "thinker_finished",
            "listener_c",
            "p2a_pop",
            "p2a",
            "p2b",
            "listener_intervention",
            "p9c",
        ),
    ),
    ("When a draft still fits (Fig 4), and who survives", ("fstar", "p8_listener")),
    (
        "Listener speculation, gated (Fig 5)",
        (
            "p3",
            "p3_dev",
            "p4",
            "p4_ci",
            "p6",
            "energy_listener",
            "p7_jms",
            "stale",
            "stale_net",
            "pass2",
        ),
    ),
    ("Thinker speculation (Table 1)", ("spec_ttfa", "spec_energy", "spec_survival")),
    ("Residency and memory", ("residency", "residency_memory", "worker_memory")),
    (
        "Thesis only: value side, triggers and the device",
        (
            "probe_greedy_050",
            "probe_greedy_075",
            "probe_greedy_090",
            "probe_greedy_gate",
            "probe_sampled_gate",
            "probe_sampled_075",
            "trigger_points",
            "trigger_tsem",
            "trigger_epa",
            "epa_wall",
            "epa_rtf",
            "llm_rate",
        ),
    ),
    (
        "Thesis only: provenance gaps closed 2026-10-14",
        (
            "attr_paired",
            "thp_m0",
            "thp_m1",
            "thp_m2",
            "thp_m3",
            "pg_first",
            "pg_survival",
            "pg_blocked",
            "carry",
            "p6_pass2",
            "pass2_dev",
            "p9_kill",
            "onset_c96",
            "onset_c64",
            "lead_one",
            "lead_shared",
            "lead_cs_shared",
            "lead_new",
            "residency_full",
        ),
    ),
    (
        "The Level-2 planner, simulated (no measured run)",
        ("planner_validation", "planner_dp", "planner_noduty"),
    ),
)

CAVEATS = {
    "stages_dev": "P1 (share rises with duration) is FALSIFIED as registered (Table 3); "
    "what stands is that the final decode is the largest stage on every set.",
    "tax_ballast_7dc19e": "Paired per turn position by occupancy_tax.py (2026-10-13, existing "
    "logs), CI resampling the 16 files. One run per condition, run after B: time order "
    "is confounded with condition.",
    "tax_ballast_ace562": "Run 5d30c2 is not a plain replicate: it added vmstat/smaps "
    "sampling around every decode. The two runs disagree on size and on the ordering "
    "of conditions; both exclude 0.",
    "thinker_overlap": "Reproduces the 2026-09-21 figure (+100.3 [+78.0, +123.9] n=130), "
    "which had no script; the paper cites this one.",
    "listener_c": "Below the registered band [500, 1500] ms/s; separate-arm coding (added "
    "post hoc, conditioned interval) gives 576 [518, 638]. Scorer written after the data.",
    "listener_intervention": "Reproduces pass 1n's +315 [+188, +453] n=16 (no script); one "
    "item dropped by the reference-match filter.",
    "p9c": "The P9b-b CI's lower end (+57 ms) lies below its 150 ms bar. Under the "
    "original check 2 (3894a8c) this result would be NOT TESTABLE; it holds under the "
    "amendment (4184a7d), and it is conditional on the 3 s post-reply quiet period.",
    "thinker_finished": "Not a general zero: the 2026-09-21 onset test found +128.8 ms "
    "[+57.6, +155.7] for finished drafts at B=64 contended (Table 3).",
    "worker_memory": "At spawn, before any decode; the per-decode working set is not logged.",
    "p2a_pop": "The bootstrap resamples the runs.",
    "carry": "The interval spans zero: carry-over is NOT established. The registered "
    "interval (+65, +1204) held the reference median fixed; this one resamples both groups.",
    "attr_paired": "Not detectable; the earlier difference-of-medians estimate lies just "
    "above the interval's upper end.",
    "onset_c96": "Timed from speech onset (tts_first_audio), not from the endpoint.",
    "onset_c64": "Timed from speech onset. Under this re-score the uncontended B=96 "
    "finished-draft interval also excludes zero.",
    "lead_cs_shared": "Both runs per configuration; NOTES's 1701 came from one run each.",
    "pass2_dev": "Pass 2's baseline (10-02) and treatment (10-05) ran on different days.",
    "epa_rtf": "Replaces the withdrawn 4.8x (11.5 / 2.4).",
    "planner_validation": "Simulation from logs, not a measurement. Its interval excludes "
    "the measured +351 on these items (it overlaps pass 2k's [+294, +412]). The final's slope is "
    "fitted largely from the contrast this line reproduces; the committed fraction is the "
    "real check (gate 0.40, ungated 0.44 simulated; measured 0.40-0.43 and 0.42-0.46).",
    "planner_dp": "Decision on f5e1b8f: no live pass; the gate is near-optimal under the "
    "measured cost model. Fill p 0.1 is the calibrated setting; with no fill the DP gains "
    "+0 [+0, +0]; richer fills raise it (+20, +23, +39 at p 0.2, 0.3, 0.57) and a "
    "steeper final slope (95.6 vs 68.2 ms/s) would scale it. Mean +92 ms, 15/70 items >= 100 "
    "ms. The adaptive-prior hypothesis is unsimulated.",
    "planner_noduty": "Decoding back to back, not planning; the duty bound is a harness "
    "constant equal across arms.",
    "stale_net": "Per-decode median 0.60 J understates the cost; mean and total are the "
    "figures to quote. Some stale decodes drew 15-17 W, unexplained (Limitations).",
    "fstar": "As registered (mixed clocks) the estimate was 0.895 [0.891, 0.901]; 0.861 is "
    "the clock-consistent correction and the figure cited. Predicted 0.858.",
    "p8_listener": "Expectation >= 0.60 not met. The thinker half is a curve value, greedy "
    "p_usable ~0.15 at the corrected f* (expectation <= 0.10).",
    "p3": "THE CI's LOWER END (+294 ms) LIES BELOW THE 300 ms BAR; reported beside the bar "
    "every time the number appears (decision on abb44e6).",
    "p4": "Mean dWER CI [-0.0039, +0.0135]; 14 items worse, 8 better.",
    "energy_listener": "Net is the like-for-like figure: the energy window ends at first "
    "audio, which STOP moves earlier.",
    "stale": "Raw figures; the net mean and total are the next claim.",
    "pass2": "Unconditional ablation: speculating on every turn, without the gate or the "
    "SIGSTOP preemption. paired = NOPARTIAL - COMMIT-WL, a reduction: -155 ms means "
    "COMMIT-WL was slower; the CI includes 0. P3 FAILS on this pass (Table 3).",
    "spec_ttfa": "Descriptive; scorer post hoc; interleave design broke at the first split "
    "(NOTES 2026-10-13). Effects below ~90 ms are not excluded.",
    "residency": "Fence-compliant headline (rep U1 dropped); full pass +12.1 [-31.1, +39.1].",
    "residency_memory": "Idle, never-decoding tiny engine: isolates residency, not the "
    "two-engine design's live memory (~+121 MB, not re-measured).",
}


def results_page() -> str:
    out = [
        "# RESULTS",
        "",
        "Every claim the paper makes, in paper order. Each is the scorer's own output",
        "line, quoted; the command reproduces it from results/raw; the NOTES entry",
        "records it, with the commit that added the entry. Generated by",
        "src/scripts/paper_results.py (`make results`). The paper is written from this",
        "page, not from the notebook. Verdicts of every registered prediction: Table 3",
        "(results/tables/paper_table3_ledger.md). Medians with 95% bootstrap CIs and n",
        "as the scorers print them; where a line carries a range, a count or a ratio",
        "instead, that is all its scorer measures, and the caveat says so.",
        "",
        "Measurement froze 2026-10-13 13:03 with pass 1r. Pass 9 (live microphone) is",
        "the approved post-freeze exception: descriptive, no claim; PENDING.",
        "",
    ]
    for title, keys in SECTIONS:
        out += [f"## {title}", ""]
        for k in keys:
            c = claim(k)
            src = ps.SOURCES[c.source]
            runs = ", ".join(src.runs) or "see the scorer's run list"
            out += [
                f"### {c.statement}",
                "",
                "```",
                c.quote(),
                "```",
                "",
                f"- command: `{src.command()}`"
                if len(src.command()) < 160
                else f"- command: `src/scripts/{src.script}` with the {len(src.runs)} runs below",
                f"- runs: {runs}",
                f'- NOTES: "{c.notes}" ({commit_of(c.notes)})',
            ]
            if k in CAVEATS:
                out.append(f"- caveat: {CAVEATS[k]}")
            out.append("")
    out += [
        "## The allocator's decisions (Table 2)",
        "",
        "Computed by this script from the decision records of "
        f"{Path(ps.PASS6[0]).name} (pass 6), not quoted from a scorer line; the "
        "values below are what the paper's macros carry.",
        "",
        "```",
        *(f"{k} = {v}" for k, v in pass6_numbers().items()),
        "```",
        "",
        '- NOTES: "2026-10-09 — PASS 8 (f*), PASS 6 (ALLOCATOR), P8. Scored once, as '
        'registered in 71d1612 + amendments"',
        "",
    ]
    return "\n".join(out)


# ------------------------------------------------------------ paper numbers --
# Every number the paper's prose cites, as a LaTeX macro taken from a claim's
# quoted line. main.tex types no number of its own.
NUMBERS: tuple[tuple[str, str, str], ...] = (
    ("PthreeRed", "p3", r"(\+\d+ ms \[[^\]]+\])"),
    ("PthreeN", "p3", r"n=(\d+)"),
    ("DevTTFA", "p3_dev", r"(\+?-?\d+ ms \[[^\]]+\])"),
    ("SttShareMulti", "stages_multi_step", r"= (\d+)% of the stage"),
    ("LlmShareMulti", "stages_multi_step", r"LLM first token \d+ ms = (\d+)%"),
    ("TTFAMulti", "stages_multi_step", r"TTFA median (\d+) ms"),
    ("SttShareDev", "stages_dev", r"= (\d+)% of the stage"),
    ("SttShareShort", "stages_short_digit", r"= (\d+)% of the stage"),
    ("SttShareLong", "stages_long_digit", r"= (\d+)% of the stage"),
    ("ThinkerOverlap", "thinker_overlap", r"endpoint: (\+[0-9.]+ ms \[[^\]]+\])"),
    ("ThinkerOverlapN", "thinker_overlap", r"n=(\d+)"),
    ("ListenerC", "listener_c", r"c (\d+) ms/s"),
    ("ListenerCCI", "listener_c", r"c \d+ ms/s (\[[^\]]+\])"),
    ("PartialsDev", "listener_intervention", r"(\+[0-9.]+ ms \[[^\]]+\])"),
    ("TaxBallastA", "tax_ballast_7dc19e", r"(\+\d+ ms \[[^\]]+\])"),
    ("TaxBallastB", "tax_ballast_ace562", r"(\+\d+ ms \[[^\]]+\])"),
    ("TaxFullA", "tax_full_37443c", r"(\+\d+ ms \[[^\]]+\])"),
    ("WordsCommitted", "p8_listener", r"median (\S+ \[[^\]]+\])"),
    ("EnergyListener", "energy_listener", r"j_net: (\+[0-9.]+ J \[[^\]]+\])"),
    ("SpecEnergy", "spec_energy", r"net SPEC - REACTIVE: (\+[0-9.]+ \[[^\]]+\])"),
    ("SpecTTFA", "spec_ttfa", r"\(ms\): (\S+ \[[^\]]+\])"),
    ("SpecTokens", "spec_survival", r"per turn median (\d+)"),
    ("Fstar", "fstar", r"f\* ([0-9.]+ \[[^\]]+\])"),
    ("PnineTTFA", "p9c", r"P9b-b paired TTFA (\+\d+ ms \[[^\]]+\])"),
    ("ResidencyTTFA", "residency", r"(\+[0-9.]+ ms \[[^\]]+\])"),
    ("WorkerPSS", "worker_memory", r"PSS median ([0-9.]+) MB"),
    ("PtwoTTFA", "pass2", r"paired\s+(-?[0-9.]+)\s+CI"),
    ("PlannerDP", "planner_dp", r"(\+\d+ ms \[[^\]]+\])"),
    ("PlannerNoDuty", "planner_noduty", r"(\+\d+ ms \[[^\]]+\])"),
    ("StaleMean", "stale_net", r"mean ([0-9.]+) J"),
    ("StaleN", "stale", r"(\d+) measured"),
    ("PfourMean", "p4", r"mean (\+[0-9.]+)"),
    ("PsixRatio", "p6", r"item-median ([0-9.]+x \[[^\]]+\])"),
    ("SevenJms", "p7_jms", r"medians (\+[0-9.]+ J/ms \[[^\]]+\])"),
    ("ResidencyMem", "residency_memory", r"medians\): \+([0-9.]+) MB"),
    ("PfourCI", "p4_ci", r"CI (\[[^\]]+\])"),
    ("PfourWorse", "p4_ci", r"worse (\d+)"),
    ("PfourBetter", "p4_ci", r"better (\d+)"),
    ("FstarUsable", "fstar", r"p_usable there ([0-9.]+)"),
    ("FstarN", "fstar", r"n=(\d+)"),
    ("WordsN", "p8_listener", r"n=(\d+)"),
    ("StaleTotal", "stale_net", r"total ([0-9.]+) J"),
    ("AloneSTT", "fig2_alone", r"STT final (\d+ ms \[[^\]]+\])"),
    ("AloneFallback", "fig2_alone", r"thp_fault_fallback (\d+)/decode"),
    ("AloneHuge", "fig2_alone", r"AnonHugePages (\d+) MB"),
    ("BallastFallback", "fig2_ballast", r"thp_fault_fallback (\d+)/decode"),
    ("BallastHuge", "fig2_ballast", r"AnonHugePages (\d+) MB"),
    ("TaxFullB", "tax_full_74e10a", r"(\+\d+ ms \[[^\]]+\])"),
    ("PnineRatio", "p9c", r"P9b-a STOP ratio ([0-9.]+x \[[^\]]+\])"),
    ("PnineN", "p9c", r"n=(\d+)"),
    ("PartialsDevN", "listener_intervention", r"n=(\d+)"),
    ("ThinkerFinished", "thinker_finished", r"endpoint: (\+[0-9.]+ ms \[[^\]]+\])"),
    ("ThinkerFinishedN", "thinker_finished", r"n=(\d+)"),
    ("SpecN", "spec_ttfa", r"n=(\d+)"),
    ("SpecItems", "spec_survival", r"n=(\d+) items"),
    ("SpecAccepted", "spec_survival", r"accepted/produced median ([0-9.]+)"),
    ("GreedySeventyFiveK", "probe_greedy_075", r"(\d+/80)"),
    ("TriggerPoints", "trigger_points", r"decision points: (\d+)"),
    ("TriggerUtts", "trigger_points", r"utterances: (\d+)"),
    ("LlmRate", "llm_rate", r"median ([0-9.]+) ms/token"),
    ("PtwoaPop", "p2a_pop", r"(\d+ turns, \d+ runs)"),
    ("AttrN", "attr_paired", r"n=(\d+)"),
    ("ThpMZero", "thp_m0", r"median (\d+)\.\d MB"),
    ("ThpMOne", "thp_m1", r"median (\d+)\.\d MB"),
    ("ThpMTwo", "thp_m2", r"median (\d+)\.\d MB"),
    ("ThpMThree", "thp_m3", r"median (\d+)\.\d MB"),
    ("PgFirst", "pg_first", r"EXACT (\d+/\d+)"),
    ("PgSurvival", "pg_survival", r"survived ([0-9.]+%)"),
    ("PgSurvivalCI", "pg_survival", r"CI over turns (\[[^\]]+\])"),
    ("PgSurvivalN", "pg_survival", r"n=(\d+) turns"),
    ("PgBlockedN", "pg_blocked", r"n=\s*(\d+)"),
    ("CarryN", "carry", r"n=(\d+)/"),
    ("PtwoSixN", "p6_pass2", r"n=(\d+)"),
    ("PtwoDevN", "pass2_dev", r"n=(\d+)"),
    ("OnsetNinetySixRun", "onset_c96", r"\| (\+[0-9.]+) \["),
    ("OnsetNinetySixRunCI", "onset_c96", r"\| \+[0-9.]+ (\[[^\]]+\])"),
    ("OnsetNinetySixRunN", "onset_c96", r"\| \S+ \[[^\]]+\] n=(\d+)"),
    ("OnsetSixtyFourFin", "onset_c64", r"32\s+(\+[0-9.]+) \["),
    ("OnsetSixtyFourFinCI", "onset_c64", r"32\s+\+[0-9.]+ (\[[^\]]+\])"),
    ("OnsetSixtyFourFinN", "onset_c64", r"\] n=(\d+) \|"),
    ("LeadOneN", "lead_one", r"n=(\d+)"),
    ("LeadSharedN", "lead_shared", r"n=(\d+)"),
    ("LeadNewN", "lead_new", r"n=(\d+)"),
    ("EpaWall", "epa_wall", r"over all \d+: ([0-9.]+) s/segment"),
    ("EpaN", "epa_wall", r"n=(\d+)"),
    ("EpaRtf", "epa_rtf", r":\s+([0-9.]+)x"),
    ("ResidencyFull", "residency_full", r"(\+[0-9.]+ ms \[[^\]]+\])"),
    ("TriggerUnindexed", "trigger_points", r"no utterance index: (\d+)"),
    ("PlannerVal", "planner_validation", r"(\+\d+ ms \[[^\]]+\])"),
    ("PtwoN", "pass2", r"n=(\d+)"),
    ("ResidencyMemRange", "residency_memory", r"range over run pairs (\+[0-9.]+ to \+[0-9.]+) MB"),
)


# Numbers that need rounding: the groups are floats, formatted as given.
FORMATTED: tuple[tuple[str, str, str, str], ...] = (
    (
        "AttrPaired",
        "attr_paired",
        r"median ([+-][0-9.]+) ms, 95% CI \[([+-][0-9.]+), ([+-][0-9.]+)\]",
        "{0:+.1f} ms [{1:+.1f}, {2:+.1f}]",
    ),
    (
        "PgBlocked",
        "pg_blocked",
        r"spec_always_pg:\s+([+-]\d+) ms\s+95% CI \[([+-]\d+), ([+-]\d+)\]",
        "{0:+.0f} ms [{1:+.0f}, {2:+.0f}]",
    ),
    (
        "Carry",
        "carry",
        r":\s+([+-]\d+) ms \[([+-]\d+), ([+-]\d+)\]",
        "{0:+.0f} ms [{1:+.0f}, {2:+.0f}]",
    ),
    (
        "PtwoSix",
        "p6_pass2",
        r"median ([0-9.]+)x CI \[([0-9.]+), ([0-9.]+)\]",
        "{0:.2f}x [{1:.2f}, {2:.2f}]",
    ),
    (
        "PtwoDev",
        "pass2_dev",
        r"paired\s+(-?[0-9.]+)\s+CI \[(-?[0-9.]+), \+?(-?[0-9.]+)\]",
        "{0:+.0f} ms [{1:+.0f}, {2:+.0f}]",
    ),
    (
        "PnineKill",
        "p9_kill",
        r"ratio\s+([0-9.]+)x \[([0-9.]+)x, ([0-9.]+)\]",
        "{0:.2f}x [{1:.2f}, {2:.2f}]",
    ),
    (
        "LeadOne",
        "lead_one",
        r"median\s+([0-9.]+) ms\s+95% CI \[\s*([0-9.]+),\s+([0-9.]+)\]",
        "{0:.0f} ms [{1:.0f}, {2:.0f}]",
    ),
    (
        "LeadShared",
        "lead_shared",
        r"median\s+([0-9.]+) ms\s+95% CI \[\s*([0-9.]+),\s+([0-9.]+)\]",
        "{0:.0f} ms [{1:.0f}, {2:.0f}]",
    ),
    (
        "LeadSharedCS",
        "lead_cs_shared",
        r"median\s+([0-9.]+) ms\s+95% CI \[\s*([0-9.]+),\s+([0-9.]+)\]",
        "{0:.0f} ms [{1:.0f}, {2:.0f}]",
    ),
    (
        "LeadNew",
        "lead_new",
        r"median\s+([0-9.]+) ms\s+95% CI \[\s*([0-9.]+),\s+([0-9.]+)\]",
        "{0:.0f} ms [{1:.0f}, {2:.0f}]",
    ),
    (
        "GreedySeventyFive",
        "probe_greedy_075",
        r"= ([0-9.]+)\s+\[([0-9.]+), ([0-9.]+)\]",
        "{0:.3f} [{1:.3f}, {2:.3f}]",
    ),
    (
        "GreedyNinety",
        "probe_greedy_090",
        r"= ([0-9.]+)\s+\[([0-9.]+), ([0-9.]+)\]",
        "{0:.3f} [{1:.3f}, {2:.3f}]",
    ),
    (
        "GreedyFifty",
        "probe_greedy_050",
        r"= ([0-9.]+)\s+\[([0-9.]+), ([0-9.]+)\]",
        "{0:.3f} [{1:.3f}, {2:.3f}]",
    ),
    (
        "SampledSeventyFive",
        "probe_sampled_075",
        r"= ([0-9.]+)\s+\[([0-9.]+), ([0-9.]+)\]",
        "{0:.3f} [{1:.3f}, {2:.3f}]",
    ),
    (
        "GreedyGate",
        "probe_greedy_gate",
        r"\s([0-9.]+)\s+80\s+\[([0-9.]+), ([0-9.]+)\]",
        "{0:.3f} [{1:.3f}, {2:.3f}]",
    ),
    (
        "SampledGate",
        "probe_sampled_gate",
        r"\s([0-9.]+)\s+80\s+\[([0-9.]+), ([0-9.]+)\]",
        "{0:.3f} [{1:.3f}, {2:.3f}]",
    ),
    (
        "TsemAUC",
        "trigger_tsem",
        r"(0\.\d+)\s+\[([0-9.]+), ([0-9.]+)\]",
        "{0:.3f} [{1:.3f}, {2:.3f}]",
    ),
    ("EpaAUC", "trigger_epa", r"(0\.\d+)\s+\[([0-9.]+), ([0-9.]+)\]", "{0:.3f} [{1:.3f}, {2:.3f}]"),
    (
        "PtwoCI",
        "pass2",
        r"paired\s+(-?[0-9.]+)\s+CI \[(-?[0-9.]+), \+?(-?[0-9.]+)\]",
        "{0:+.0f} ms [{1:+.0f}, {2:+.0f}]",
    ),
    (
        "SpecTTFAr",
        "spec_ttfa",
        r"\(ms\): (\S+) \[(\S+), (\S+)\]",
        "{0:+.0f} ms [{1:+.0f}, {2:+.0f}]",
    ),
    (
        "SpecEnergyr",
        "spec_energy",
        r"REACTIVE: (\S+) \[(\S+), (\S+)\]",
        "{0:+.1f} J [{1:+.1f}, {2:+.1f}]",
    ),
    (
        "EnergyListenerr",
        "energy_listener",
        r"j_net: (\S+) J \[(\S+), (\S+)\]",
        "{0:+.1f} J [{1:+.1f}, {2:+.1f}]",
    ),
)


def formatted(macro: str) -> str:
    """The FORMATTED value of macro, rounded as the paper prints it."""
    for name, key, pat, fmt in FORMATTED:
        if name == macro:
            m = re.search(pat, claim(key).quote())
            if not m:
                raise LookupError(f"{key}: {pat!r}")
            return fmt.format(*(float(g) for g in m.groups())).replace("+-", "-")
    raise LookupError(macro)


def pass6_numbers() -> dict[str, str]:
    (run,) = ps.PASS6
    recs = [
        json.loads(x)
        for x in (REPO / run / "turns.jsonl").read_text().splitlines()
        if '"decision_record"' in x
    ]
    reasons = Counter(r["decision"]["inputs"]["reason"] for r in recs)
    be = {r["decision"]["inputs"]["break_even"] for r in recs}
    if len(be) != 1:
        raise ValueError(f"break-even not constant: {be}")
    return {
        "AllocDecisions": str(len(recs)),
        "AllocBelow": str(reasons["below_break_even"]),
        "AllocInfeasible": str(reasons["infeasible"]),
        "AllocSpends": str(sum(1 for r in recs if r["decision"]["budget_tokens"] > 0)),
        "BreakEven": f"{be.pop():.4f}",
    }


def numbers_tex() -> str:
    out = ["% Generated by src/scripts/paper_results.py from the claim registry.", "% Do not edit."]
    for macro, key, pat in NUMBERS:
        val = grab(key, pat).replace("%", r"\%")
        out.append(f"\\newcommand{{\\{macro}}}{{{val}}}  % {key}")
    for macro, key, _pat, _fmt in FORMATTED:
        val = formatted(macro)
        out.append(f"\\newcommand{{\\{macro}}}{{{val}}}  % {key}")
    for macro, val in pass6_numbers().items():
        out.append(f"\\newcommand{{\\{macro}}}{{{val}}}  % pass 6 decision records")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------ LaTeX tables --
def tex(x: str) -> str:
    x = re.sub(
        r"[\\{}]", lambda m: {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}"}[m.group()], x
    )
    for a_, b_ in (
        ("~", r"\textasciitilde{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        (">=", r"$\ge$"),
        ("<=", r"$\le$"),
        ("->", r"$\to$"),
        ("--parallel", r"\texttt{-{}-parallel}"),
    ):
        x = x.replace(a_, b_)
    x = re.sub(r"(?<![A-Za-z])'([^']+)'(?![A-Za-z])", r"`\1'", x)
    return re.sub(r'"([^"]+)"', r"``\1''", x)


def table1_tex() -> str:
    out = [
        "% Generated by src/scripts/paper_results.py. Do not edit.",
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\caption{Where to speculate: the listener against the thinker on this device. "
        r"Every cell is a scorer's output; sources in RESULTS.md.}",
        r"\label{tab:where}",
        r"\begin{tabular}{p{0.15\textwidth}p{0.42\textwidth}p{0.36\textwidth}}",
        r"\toprule",
        r" & Listener (COMMIT-WL-STOP + gate) & Thinker (SPEC-CONTINUE) \\",
        r"\midrule",
    ]
    out += [f"{tex(a_)} & {tex(b_)} & {tex(c_)} \\\\" for a_, b_, c_ in table1_rows()]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


def table2_tex() -> str:
    md = [ln for ln in table2().splitlines() if ln.startswith("| ") and "---" not in ln]
    head, body = md[0], md[1:]
    cols = [c.strip() for c in head.strip("|").split("|")]
    out = [
        "% Generated by src/scripts/paper_results.py. Do not edit.",
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\caption{The allocator's decisions in one multi-step run, by reason: median "
        r"(min to max) of the inputs each decision logged.}",
        r"\label{tab:allocator}",
        r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}",
        r"\toprule",
        " & ".join(tex(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for ln in body:
        out.append(" & ".join(tex(c.strip()) for c in ln.strip("|").split("|")) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


def table3_tex() -> str:
    ledger = json.loads(LEDGER.read_text())
    out = [
        "% Generated by src/scripts/paper_results.py. Do not edit.",
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\caption{Every pre-registered prediction: the commit that registered it, the "
        r"commit that scored it, and its verdict as it stands after all corrections. The "
        r"full ledger, with bars, figures and notes, is in the supplementary material.}",
        r"\label{tab:ledger}",
        r"\begin{tabular}{lllp{0.25\textwidth}p{0.36\textwidth}}",
        r"\toprule",
        r"prediction & registered & scored & verdict as it stands & qualifier \\",
        r"\midrule",
    ]
    for e in ledger:
        reg = commit_of(e["registered"]).split()[0]
        sc = commit_of(e["scored"]).split()[0] if e.get("scored") else "--"
        head = e["verdict"].split(":")[0].strip()
        q = tex(e.get("qualifier", ""))
        out.append(
            f"{tex(e['id'])} & \\texttt{{{reg}}} & \\texttt{{{sc}}} & {tex(head)} & {q} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


# The Interspeech paper's Table 3: six rows (Ali, decision on 917c10f), each the
# ledger's verdict head and one qualifier; the full ledger is the extended version.
SUMMARY: tuple[tuple[str, str, str, str], ...] = (
    (
        "P1",
        "P1",
        "final's share of TTFA >= 50% at every length, rising with length",
        "the final is still the largest stage on every set; its share is not monotone in length",
    ),
    (
        "P2a",
        "P2a",
        "listener overlap cost 0.5 to 1.5 s of final per s of overlap",
        "scorer written after the data",
    ),
    (
        "P3 (pass 2k)",
        "P3 (pass 2k)",
        "gated listener cuts median TTFA by >= 300 ms on multi-step",
        "CI lower end below the 300 ms bar; the ungated test of pass 2 FAILED",
    ),
    (
        "P4 (pass 2)",
        "P4",
        "dWER <= +0.020 on every set",
        "the selection-set estimate did not carry; also HOLDS in pass 2k",
    ),
    (
        "P7 (pass 2)",
        "P7",
        "listener energy per turn no higher than reactive",
        "restated per ms saved in pass 2k",
    ),
    (
        "P9c",
        "P9c",
        "freezing the orphan saves >= 150 ms of TTFA",
        "CI lower end below the bar; holds under a pre-run amendment; needs the 3 s quiet period",
    ),
)


def table3_summary_tex() -> str:
    ledger = {e["id"]: e for e in json.loads(LEDGER.read_text())}
    out = [
        "% Generated by src/scripts/paper_results.py. Do not edit.",
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\caption{Six of the " + str(len(ledger)) + r" entries of the pre-registration "
        r"ledger, with their verdicts as they stand and one qualifier each. The full ledger, with "
        r"registration and scoring commits, is in the extended version.}",
        r"\label{tab:ledger}",
        r"\begin{tabular}{p{0.11\textwidth}p{0.27\textwidth}p{0.2\textwidth}p{0.32\textwidth}}",
        r"\toprule",
        r"prediction & predicted & verdict & qualifier \\",
        r"\midrule",
    ]
    for label, key, pred, q in SUMMARY:
        head = ledger[key]["verdict"].split(":")[0].strip()
        out.append(f"{tex(label)} & {tex(pred)} & {tex(head)} & {tex(q)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------- thesis appendices --
def ledger_full_tex() -> str:
    """Appendix: the whole ledger, every field, as a longtable."""
    ledger = json.loads(LEDGER.read_text())
    out = [
        "% Generated by src/scripts/paper_results.py. Do not edit.",
        r"\begin{footnotesize}",
        r"\begin{longtable}{p{0.13\textwidth}p{0.26\textwidth}p{0.11\textwidth}p{0.4\textwidth}}",
        r"\caption{The pre-registration ledger: every entry, its registration and "
        r"scoring commits, its verdict as it stands, corrections and notes.}"
        r"\label{tab:ledger-full}\\",
        r"\toprule",
        r"entry & prediction & commits & verdict, corrections, notes \\",
        r"\midrule\endfirsthead",
        r"\toprule entry & prediction & commits & verdict, corrections, notes \\ "
        r"\midrule\endhead",
        r"\bottomrule\endfoot",
    ]
    for e in ledger:
        reg = commit_of(e["registered"]).split()
        commits = f"reg.\\ \\texttt{{{reg[0]}}} ({reg[1]})"
        if e.get("registered_original"):
            commits = (
                f"orig.\\ \\texttt{{{commit_of(e['registered_original']).split()[0]}}}; " + commits
            )
        if e.get("scored"):
            commits += f"; scored \\texttt{{{commit_of(e['scored']).split()[0]}}}"
        cell = tex(e["verdict"])
        corr = [commit_of(h).split()[0] for h in e.get("corrections", [])]
        if corr:
            cell += r" \emph{Corrections:} " + ", ".join(f"\\texttt{{{c}}}" for c in corr) + "."
        for k in ("qualifier", "notes"):
            if e.get(k):
                cell += r" \emph{" + k.capitalize() + ":} " + tex(e[k])
        out.append(f"{tex(e['id'])} & {tex(e['prediction'])} & {commits} & {cell} \\\\")
    out += [r"\end{longtable}", r"\end{footnotesize}"]
    return "\n".join(out) + "\n"


RETRACTION = re.compile(r"CORRECTION|RETRACTION|WITHDRAWN|\bVOID\b|VOIDED|PROVENANCE GAP")
NOT_RETRACTION = re.compile(r"EXPECTED TO VOID|RE-RUN: check 1 voids")


def retractions_tex() -> str:
    """Appendix: every notebook entry that corrects, retracts, voids or closes a
    provenance gap, in order, with the commit that added it."""
    heads = [
        ln[3:].strip()
        for ln in (REPO / NOTES).read_text().splitlines()
        if ln.startswith("## ") and RETRACTION.search(ln) and not NOT_RETRACTION.search(ln)
    ]
    out = [
        "% Generated by src/scripts/paper_results.py. Do not edit.",
        r"\begin{footnotesize}",
        r"\begin{longtable}{p{0.1\textwidth}p{0.72\textwidth}l}",
        r"\caption{Retractions log: every notebook entry whose heading marks a "
        r"correction, a retraction, a withdrawal or a voided run, in order, with the "
        r"commit that added it. The corrected entries stand unedited beside them; "
        r"retractions stated inside other entries are recorded in the ledger's "
        r"notes.}\label{tab:retractions}\\",
        r"\toprule",
        r"\# & notebook entry & commit \\",
        r"\midrule\endfirsthead",
        r"\toprule \# & notebook entry & commit \\ \midrule\endhead",
        r"\bottomrule\endfoot",
    ]
    for i, h in enumerate(heads, 1):
        c = commit_of(h).split()
        out.append(f"{i} & {tex(h)} & \\texttt{{{c[0]}}} {c[1]} \\\\")
    out += [r"\end{longtable}", r"\end{footnotesize}"]
    return "\n".join(out) + "\n"


def md_to_tex(md: str) -> str:
    """Markdown tables and paragraphs (as make_tables.py writes them) to LaTeX."""
    out: list[str] = ["% Generated by src/scripts/paper_results.py from a make_tables.py output."]
    rows: list[list[str]] = []

    def flush() -> None:
        if not rows:
            return
        cols = len(rows[0])
        out.append(r"\begin{center}\footnotesize")
        out.append(r"\begin{tabular}{" + "l" * cols + "}")
        out.append(r"\toprule")
        out.append(" & ".join(tex(c) for c in rows[0]) + r" \\")
        out.append(r"\midrule")
        out.extend(" & ".join(tex(c) for c in r) + r" \\" for r in rows[1:])
        out.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
        rows.clear()

    for ln in md.splitlines():
        if ln.startswith("|"):
            if set(ln.replace("|", "").strip()) <= set("-: "):
                continue
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
            continue
        flush()
        if ln.startswith("#"):
            title = ln.lstrip("#").strip()
            if ln.startswith("# "):
                continue  # the appendix supplies its own title
            out.append(r"\subsection*{" + tex(title) + "}")
        elif ln.strip():
            out.append(tex(ln.lstrip("- ").strip().replace("`", "")) + "\n")
    flush()
    return "\n".join(out) + "\n"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, fn in (
        ("paper_table1_where_to_speculate.md", table1),
        ("paper_table2_allocator.md", table2),
        ("paper_table3_ledger.md", table3),
    ):
        (TABLES / name).write_text(fn())
        print(f"  wrote results/tables/{name}")
    (REPO / "results/RESULTS.md").write_text(results_page())
    print("  wrote results/RESULTS.md")
    for gdir in ("paper/generated", "thesis/generated"):
        gen = REPO / gdir
        gen.mkdir(parents=True, exist_ok=True)
        _write_generated(gen)
    print("  wrote paper/generated and thesis/generated")
    gen = REPO / "thesis/generated"
    (gen / "ledger_full.tex").write_text(ledger_full_tex())
    (gen / "retractions.tex").write_text(retractions_tex())
    (gen / "capture_hazards.tex").write_text(
        md_to_tex((REPO / "results/tables/capture_hazards.md").read_text())
    )
    print("  wrote thesis/generated appendices")


def _write_generated(gen: Path) -> None:
    (gen / "numbers.tex").write_text(numbers_tex())
    for name, fn in (
        ("table1.tex", table1_tex),
        ("table2.tex", table2_tex),
        ("table3.tex", table3_tex),
        ("table3_summary.tex", table3_summary_tex),
    ):
        (gen / name).write_text(fn())


if __name__ == "__main__":
    main()
