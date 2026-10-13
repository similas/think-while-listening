"""Where every number in the paper comes from: one registry, read by the tables
and by results/RESULTS.md.

A SOURCE is one scorer invocation over named runs, run as a subprocess exactly
as a reader would type it. A CLAIM is one line of that scorer's output, found
by a pattern that must match exactly once, plus the NOTES heading that records
it. Nothing here computes a number: a claim's value is the scorer's own line,
quoted. A pattern that stops matching, or matches twice, is an error -- the
paper must not silently cite a line that has moved.

Run ids are those NOTES gives for each pass; every invocation below was checked
to reproduce the headline its entry records (2026-10-13).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = "results/raw/reactive"
MULTI = "results/raw/spoken_mqa/multi_step_reasoning-seed20260922"
DEV = "results/raw/audio/sixteen"


def _runs(*ids: str) -> tuple[str, ...]:
    return tuple(f"{RAW}/reactive-{i}" for i in ids)


@dataclass(frozen=True)
class Source:
    script: str
    args: tuple[str, ...] = ()

    @property
    def runs(self) -> tuple[str, ...]:
        return tuple(Path(a).name for a in self.args if a.startswith(RAW))

    def command(self) -> str:
        return " ".join(["src/scripts/" + self.script, *self.args])


P2K_MULTI_STOP = _runs("20261008-103300-abd218", "20261008-121644-d2b5ad", "20261008-125425-f94b44")
P2K_MULTI_NOP = _runs("20261008-110545-1f7989", "20261008-114351-9ae173", "20261008-132718-54285c")
P2K_DEV_STOP = _runs("20261008-113854-585574", "20261008-125153-9aab0d", "20261008-140024-eda306")
P2K_DEV_NOP = _runs("20261008-114121-6c121f", "20261008-124928-58fcc2", "20261008-140252-12b836")
PASS6 = _runs("20261009-120309-a4f48d")
PASS8 = _runs("20261009-105401-6ac01d", "20261009-112837-d4ef5e")

SOURCES: dict[str, Source] = {
    "figs": Source("paper_figures.py", ("--only", "1,2", "--print-only")),
    "tax": Source("occupancy_tax.py"),
    "p2": Source("p2_score.py"),
    "overlap": Source("overlap_penalties.py"),
    "p2k": Source(
        "p2k_score.py",
        (
            *("--set", "multi_step", MULTI, *P2K_MULTI_STOP, *P2K_MULTI_NOP),
            *("--set", "dev", DEV, *P2K_DEV_STOP, *P2K_DEV_NOP),
        ),
    ),
    "pass2": Source(
        "pass2_score.py",
        (
            *("--set", "multi_step", "--set-dir", MULTI, "--base"),
            *_runs("20261002-104047-106290", "20261002-112138-a6c31a", "20261002-120244-4be4d4"),
            "--test",
            *_runs("20261005-145416-dac216", "20261005-153526-e45863", "20261005-161631-1f1168"),
        ),
    ),
    "p9c": Source(
        "p9c_score.py",
        _runs(
            "20261007-190211-8058b3",
            "20261007-191402-ee2912",
            "20261007-192552-827254",
            "20261007-190807-fdf53c",
            "20261007-191955-5cd9f4",
            "20261007-193147-91df38",
        ),
    ),
    "fstar": Source("fstar_score.py", (*PASS8, "--p8", *PASS6, *P2K_MULTI_NOP)),
    "p7": Source("p7_score.py", (*_runs("20261009-141221-943340"), MULTI)),
    # 1r: the fence-compliant headline drops rep U1 (eecf2d); the full pass is
    # the sensitivity and carries the run-level memory figure the CORRECTION cites.
    "p1r": Source(
        "p1r_score.py",
        (
            DEV,
            *_runs(
                "20261013-125053-6884cf",
                "20261013-125318-3e3ca7",
                "20261013-125544-291019",
                "20261013-125815-414699",
                "20261013-130040-def96a",
            ),
        ),
    ),
    "p1r_full": Source(
        "p1r_score.py",
        (
            DEV,
            *_runs(
                "20261013-124828-eecf2d",
                "20261013-125053-6884cf",
                "20261013-125318-3e3ca7",
                "20261013-125544-291019",
                "20261013-125815-414699",
                "20261013-130040-def96a",
            ),
        ),
    ),
}


@cache
def output(key: str) -> str:
    src = SOURCES[key]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    res = subprocess.run(
        [sys.executable, str(REPO / "src/scripts" / src.script), *src.args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"{src.command()} exited {res.returncode}:\n{res.stderr[-2000:]}")
    return res.stdout


def line(key: str, pattern: str) -> re.Match[str]:
    """The one output line of SOURCES[key] matching pattern; error otherwise."""
    hits = [m for ln in output(key).splitlines() if (m := re.search(pattern, ln))]
    if len(hits) != 1:
        raise LookupError(f"{key}: {pattern!r} matched {len(hits)} lines, need exactly 1")
    return hits[0]


@dataclass(frozen=True)
class Claim:
    key: str
    statement: str
    source: str
    pattern: str
    notes: str  # the NOTES heading that records it, without "## "

    def quote(self) -> str:
        return line(self.source, self.pattern).string.strip()


H_P1N = (
    "2026-10-02 — PASS 1n COMPLETE. The partial-decode cost is real, smaller than predicted, "
    "and scales with in-flight rate"
)
H_ATTR = "2026-10-13 — THE OCCUPANCY TAX, PAIRED: intervals for the 2026-09-16 medians"
H_THP = H_ATTR
TAX_BALLAST = (
    "Occupancy tax: STT final decode with an inert process that only holds pages "
    "co-resident, minus the recognizer alone, paired per turn"
)
TAX_FULL = (
    "Occupancy tax: STT final decode with the full pipeline co-resident, minus the "
    "recognizer alone, paired per turn"
)

CLAIMS: tuple[Claim, ...] = (
    Claim(
        "p3",
        "Listener speculation done right (COMMIT-WL-STOP + feasibility gate) cuts median "
        "TTFA on multi_step against REACTIVE-NOPARTIAL",
        "p2k",
        r"^\s*P3 TTFA reduction",
        "2026-10-08 — PASS 2k: P3 HOLDS. +357 ms on multi_step, inside the predicted "
        "bracket; the gate abstains on dev",
    ),
    Claim(
        "p3_dev",
        "On dev (short utterances) the gate abstains and TTFA is unchanged",
        "p2k",
        r"^\s*DEV TTFA NOPARTIAL-STOP",
        "2026-10-08 — PASS 2k: P3 HOLDS. +357 ms on multi_step, inside the predicted "
        "bracket; the gate abstains on dev",
    ),
    Claim(
        "p4",
        "Speculative commitment costs no accuracy on multi_step (dWER)",
        "p2k",
        r"^\s*P4 dWER STOP-NOPARTIAL: .* n=70",
        "2026-10-08 — PASS 2k: P3 HOLDS. +357 ms on multi_step, inside the predicted "
        "bracket; the gate abstains on dev",
    ),
    Claim(
        "p6",
        "The final decode after a committed prefix is no slower than a solo decode of its tail",
        "p2k",
        r"^\s*P6 STOP final / solo model: .* n=70",
        "2026-10-08 — PASS 2k: P3 HOLDS. +357 ms on multi_step, inside the predicted "
        "bracket; the gate abstains on dev",
    ),
    Claim(
        "energy_listener",
        "Listener speculation's energy per turn, net of the board floor (multi_step)",
        "p2k",
        r"paired energy per turn STOP - NOPARTIAL j_net: .* n=70",
        '2026-10-08 — CORRECTION to "PASS 2k: P3 HOLDS" (reviewer, 7 findings)',
    ),
    Claim(
        "p7_jms",
        "Energy price of the latency saved, net (P7)",
        "p2k",
        r"^\s*P7 j_net:",
        '2026-10-08 — CORRECTION to "PASS 2k: P3 HOLDS" (reviewer, 7 findings)',
    ),
    Claim(
        "stale",
        "Cost of the preemption design: resumed-and-discarded stale decodes",
        "p2k",
        r"^\s*STALE DECODE COST:",
        "2026-10-08 — PASS 2k: P3 HOLDS. +357 ms on multi_step, inside the predicted "
        "bracket; the gate abstains on dev",
    ),
    Claim(
        "pass2",
        "Unconditional listener speculation (pass 2, COMMIT-WL vs NOPARTIAL) does not cut "
        "TTFA on multi_step",
        "pass2",
        r"^\s*ttfa_ms\s*:",
        "2026-10-06 — PASS 2 SCORED. P3 FAILS, P7 FALSIFIED, P4 PASSES, P6 NOT MET ON TWO SETS",
    ),
    Claim(
        "p9c",
        "Freezing the orphaned hypothesis decode at the endpoint (SIGSTOP) saves TTFA",
        "p9c",
        r"^PRIMARY:",
        "2026-10-08 — P9c HOLDS. Freezing the orphan at the endpoint saves 462 ms of TTFA, "
        "all of it in the final",
    ),
    Claim(
        "listener_c",
        "Listener overlap penalty c: final-decode ms per second of endpoint overlap (P2a)",
        "p2",
        r"^\s*final_ms = a \+ b\*audio_s \+ c\*overlap_s",
        '2026-10-13 — CORRECTION to "P2a AND P2b SCORED"',
    ),
    Claim(
        "p2a",
        "P2a verdict",
        "p2",
        r"^\s*P2a \(c in",
        '2026-10-13 — CORRECTION to "P2a AND P2b SCORED"',
    ),
    Claim(
        "p2b",
        "P2b verdict",
        "p2",
        r"^\s*P2b: NOT TESTABLE",
        '2026-10-13 — CORRECTION to "P2a AND P2b SCORED"',
    ),
    Claim(
        "listener_intervention",
        "Removing partial decodes on dev (canonical REACTIVE - REACTIVE-NOPARTIAL, pass 1n)",
        "overlap",
        r"^\s*\+[0-9.]+ ms \[.*\] n=15$",
        "2026-10-13 — PROVENANCE GAP closed: two claim-bearing numbers had no script",
    ),
    Claim(
        "thinker_overlap",
        "Thinker overlap penalty: TTFA when a draft is still decoding at the endpoint",
        "overlap",
        r"decode still running at the endpoint:",
        "2026-10-13 — PROVENANCE GAP closed: two claim-bearing numbers had no script",
    ),
    Claim(
        "thinker_finished",
        "No thinker penalty when the draft finished before the endpoint",
        "overlap",
        r"decode finished\s+at the endpoint:",
        "2026-10-13 — PROVENANCE GAP closed: two claim-bearing numbers had no script",
    ),
    Claim(
        "fstar",
        "f*, the last utterance fraction at which a draft still fits, on the buffer clock "
        "(predicted 0.858)",
        "fstar",
        r"clock=buffer emitted_only=False",
        '2026-10-09 — CORRECTION to "PASS 8 (f*), PASS 6, P8": pass 8\'s f* was on mixed '
        "clocks. Corrected, it lands on the prediction",
    ),
    Claim(
        "p8_listener",
        "Share of words the listener commits before the endpoint (P8, expected >= 0.60)",
        "fstar",
        r"^P8 LISTENER",
        '2026-10-09 — CORRECTION to "PASS 8 (f*), PASS 6, P8": pass 8\'s f* was on mixed '
        "clocks. Corrected, it lands on the prediction",
    ),
    Claim(
        "spec_ttfa",
        "Thinker speculation (SPEC-CONTINUE) vs REACTIVE, TTFA (pass 7, descriptive)",
        "p7",
        r"TTFA SPEC - REACTIVE",
        '2026-10-13 — CORRECTION to "PASS 7 SCORED" (reviewer, 4 findings)',
    ),
    Claim(
        "spec_energy",
        "Thinker speculation's energy per turn, net (pass 7, descriptive)",
        "p7",
        r"J/turn net SPEC - REACTIVE",
        '2026-10-13 — CORRECTION to "PASS 7 SCORED" (reviewer, 4 findings)',
    ),
    Claim(
        "spec_survival",
        "Thinker draft survival: tokens accepted of tokens produced (pass 7)",
        "p7",
        r"SPEC-CONTINUE within arm:",
        '2026-10-13 — CORRECTION to "PASS 7 SCORED" (reviewer, 4 findings)',
    ),
    Claim(
        "residency",
        "Keeping the second STT engine resident costs no detectable TTFA (pass 1r, U1 dropped)",
        "p1r",
        r"^PAIRED TTFA",
        '2026-10-13 — CORRECTION to "PASS 1r SCORED" (reviewer, 4 findings)',
    ),
    Claim(
        "residency_memory",
        "Memory of the resident, never-decoding tiny engine, run level (pass 1r, full pass)",
        "p1r_full",
        r"^RUN-LEVEL agent RSS delta",
        '2026-10-13 — CORRECTION to "PASS 1r SCORED" (reviewer, 4 findings)',
    ),
    Claim(
        "stages_dev",
        "Stage decomposition, REACTIVE-NOPARTIAL, dev (16)",
        "figs",
        r"Fig1 dev \(16\):",
        H_P1N,
    ),
    Claim(
        "stages_short_digit",
        "Stage decomposition, REACTIVE-NOPARTIAL, short_digit (20)",
        "figs",
        r"Fig1 short_digit \(20\):",
        H_P1N,
    ),
    Claim(
        "stages_long_digit",
        "Stage decomposition, REACTIVE-NOPARTIAL, long_digit (20)",
        "figs",
        r"Fig1 long_digit \(20\):",
        H_P1N,
    ),
    Claim(
        "stages_multi_step",
        "Stage decomposition, REACTIVE-NOPARTIAL, multi_step (80)",
        "figs",
        r"Fig1 multi_step \(80\):",
        H_P1N,
    ),
    Claim(
        "stages_longest",
        "Stage decomposition, REACTIVE-NOPARTIAL, 20 longest (20)",
        "figs",
        r"Fig1 20 longest \(20\):",
        H_P1N,
    ),
    Claim(
        "tax_ballast_7dc19e",
        TAX_BALLAST,
        "tax",
        r"TAX C' inert ballast - B.*105309-7dc19e$",
        H_ATTR,
    ),
    Claim(
        "tax_ballast_ace562",
        TAX_BALLAST,
        "tax",
        r"TAX C' inert ballast - B.*120043-ace562$",
        H_THP,
    ),
    Claim(
        "tax_full_37443c",
        TAX_FULL,
        "tax",
        r"TAX D full pipeline - B.*110333-37443c$",
        H_ATTR,
    ),
    Claim(
        "tax_full_74e10a",
        TAX_FULL,
        "tax",
        r"TAX D full pipeline - B.*121140-74e10a$",
        H_THP,
    ),
)
