"""The paper's five figures, rebuilt from raw logs by `make figures`.

Every plotted number is computed here from results/raw/, or by a named script
in src/scripts/ that this file imports; run ids are listed below, each with the
NOTES heading it belongs to. Nothing is typed in except the registered
PREDICTIONS (marked as such) and the probe's measured p_usable curve, which
lives in twl.policies with its citation.

    Fig 1  stage decomposition of TTFA, five sets, REACTIVE-NOPARTIAL
    Fig 2  the occupancy tax and its mechanism (THP fallback)
    Fig 3  the two overlap penalties on one axis
    Fig 4  p_usable(f) against window(f); f* predicted and measured
    Fig 5  pass 2k against NOPARTIAL, with pass 2 as the unconditional ablation

Style: vector PDF + PNG, Okabe-Ito colours, units on every axis, n on every bar.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twl.policies import (
    BREAK_EVEN_P_USABLE,
    MULTI_STEP_GREEDY_P_USABLE,
    window_model_ms,
)
from twl.records import endpoint_after_final

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "results/raw/reactive"
SPOKEN = REPO / "results/raw/spoken_mqa"
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def _script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def boot(v: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = sorted(statistics.median([v[rng.randrange(len(v))] for _ in v]) for _ in range(reps))
    return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]


def turns_of(run: str) -> list[dict]:
    return [
        json.loads(x)
        for x in (RAW / run / "turns.jsonl").read_text().splitlines()
        if x.strip() and '"turn_record"' in x
    ]


def scored(turns: list[dict]) -> list[dict]:
    """Valid, not split, not endpoint_after_final (the standing exclusions)."""
    return [
        t
        for t in turns
        if t.get("valid")
        and "vad_split" not in (t.get("invalid_reason") or "")
        and not endpoint_after_final(t.get("stages_ms") or {})
    ]


def mean_ci(v: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float, float, int]:
    """Mean with a percentile bootstrap CI: for dWER, whose median is 0 by
    construction (most items transcribe identically) and would hide the rest."""
    if len(v) < 2:
        return (float("nan"), float("nan"), float("nan"), len(v))
    rng = random.Random(seed)
    ms = sorted(statistics.fmean(v[rng.randrange(len(v))] for _ in v) for _ in range(reps))
    return statistics.fmean(v), ms[int(0.025 * reps)], ms[int(0.975 * reps) - 1], len(v)


PRINT_ONLY = False  # --print-only: report the plotted values, write no file


def save(fig: Any, out: Path, name: str) -> None:
    if PRINT_ONLY:
        plt.close(fig)
        return
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


# ---------------------------------------------------------------- Fig 1 ----
# REACTIVE-NOPARTIAL. Four sets from pass 1n (NOTES "PRE-REGISTRATION
# 2026-10-01e -- pass 1n"); the fifth, the 20 longest multi_step items, from the
# P5 rerun's NOPARTIAL arm (NOTES "P5 SCORED AND FAILED"), since no NOPARTIAL
# run exists on the selection set.
FIG1_SETS = {
    "dev (16)": (
        "reactive-20261002-103129-d917b8",
        "reactive-20261002-111217-ca59d8",
        "reactive-20261002-115311-72786f",
    ),
    "short_digit (20)": (
        "reactive-20261002-103332-b34440",
        "reactive-20261002-111420-86da6c",
        "reactive-20261002-115514-ed6f32",
    ),
    "long_digit (20)": (
        "reactive-20261002-103659-426fbe",
        "reactive-20261002-111804-8a8ad0",
        "reactive-20261002-115855-7562bf",
    ),
    "multi_step (80)": (
        "reactive-20261002-104047-106290",
        "reactive-20261002-112138-a6c31a",
        "reactive-20261002-120244-4be4d4",
    ),
    "20 longest (20)": ("reactive-20261006-110228-2e8de2",),
}
STAGES = (
    ("VAD settle", "speech_end_est", "vad_user_stopped"),
    ("STT final", "vad_user_stopped", "stt_final"),
    ("LLM first token", "stt_final", "llm_first_token"),
    ("TTS first chunk", "llm_first_token", "tts_first_audio"),
    ("audio out", "tts_first_audio", "audio_out_first"),
)


def fig1(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels, bottoms = [], None
    rows = []
    for name, runs in FIG1_SETS.items():
        ts = [t for r in runs for t in scored(turns_of(r))]
        stage_meds = []
        for _lab, a, b in STAGES:
            v = [
                t["stages_ms"][b] - t["stages_ms"][a]
                for t in ts
                if a in t["stages_ms"] and b in t["stages_ms"]
            ]
            stage_meds.append(statistics.median(v) if v else 0.0)
        ttfa = [
            t["stages_ms"]["audio_out_first"] - t["stages_ms"]["speech_end_est"]
            for t in ts
            if "audio_out_first" in t["stages_ms"] and "speech_end_est" in t["stages_ms"]
        ]
        speech = statistics.median(
            t["stages_ms"]["speech_end_est"] for t in ts if "speech_end_est" in t["stages_ms"]
        )
        rows.append((name, stage_meds, statistics.median(ttfa), len(ts), speech / 1000))
        print(
            f"  Fig1 {name}: TTFA median {statistics.median(ttfa):.0f} ms, STT final "
            f"{stage_meds[1]:.0f} ms = {stage_meds[1] / sum(stage_meds) * 100:.0f}% of the stage "
            f"sum, LLM first token {stage_meds[2]:.0f} ms = "
            f"{stage_meds[2] / sum(stage_meds) * 100:.0f}%, n={len(ts)} turns, {len(runs)} runs"
        )
    x = range(len(rows))
    bottoms = [0.0] * len(rows)
    for k, (lab, _a, _b) in enumerate(STAGES):
        vals = [r[1][k] for r in rows]
        ax.bar(x, vals, bottom=bottoms, color=OKABE[k], label=lab, width=0.62)
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
    for i, r in enumerate(rows):
        stt_share = r[1][1] / sum(r[1]) * 100
        ax.text(
            i,
            bottoms[i] + 60,
            f"TTFA {r[2]:.0f}\nSTT {stt_share:.0f}%\nn={r[3]}",
            ha="center",
            fontsize=7,
        )
        labels.append(f"{r[0]}\nspeech {r[4]:.1f} s")
    ax.set_xticks(list(x), labels, fontsize=7)
    ax.set_ylabel("median stage duration after end of speech (ms)")
    ax.set_ylim(0, max(bottoms) * 1.3)
    ax.legend(fontsize=7, ncol=3, loc="upper left", frameon=False)
    ax.set_title("Fig 1. What the silence is made of (REACTIVE-NOPARTIAL)", fontsize=9)
    fig.tight_layout()
    save(fig, out, "paper_fig1_stage_decomposition")


# ---------------------------------------------------------------- Fig 2 ----
# The six per-condition runs of stt-attribution-20260916-115211-5d30c2 (NOTES
# "2026-09-16 -- The sub-mechanism: THP fallback, not reclaim").
FIG2_CONDITIONS = (
    ("B: recognizer alone", "reactive-20260916-115218-058b62"),
    ("C': inert ballast", "reactive-20260916-120043-ace562"),
    ("C''b: llama, no CUDA", "reactive-20260916-120813-52c75a"),
    ("C: llama idle", "reactive-20260916-115628-2f3a6e"),
    ("C''a: CUDA ctx, ngl 0", "reactive-20260916-120420-ba6fc4"),
    ("D: full pipeline", "reactive-20260916-121140-74e10a"),
)


def fig2(out: Path) -> None:
    rows = []
    for label, run in FIG2_CONDITIONS:
        ts = [t for t in turns_of(run) if t.get("valid")]
        stt = [
            t["stages_ms"]["stt_final"] - t["stages_ms"]["vad_user_stopped"]
            for t in ts
            if "stt_final" in t["stages_ms"] and "vad_user_stopped" in t["stages_ms"]
        ]
        fb = [float((t.get("vmstat_delta") or {}).get("thp_fault_fallback", 0)) for t in ts]
        huge = [
            float(t.get("anon_huge_after_mb", -1))
            for t in ts
            if t.get("anon_huge_after_mb", -1) >= 0
        ]
        rows.append(
            (
                label,
                statistics.median(stt),
                boot(stt),
                statistics.median(fb),
                statistics.median(huge),
                len(stt),
            )
        )
        r = rows[-1]
        print(
            f"  Fig2 {label}: STT final {r[1]:.0f} ms [{r[2][0]:.0f}, {r[2][1]:.0f}] n={r[5]}; "
            f"thp_fault_fallback {r[3]:.0f}/decode; AnonHugePages {r[4]:.0f} MB; run {run}"
        )
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    x = list(range(len(rows)))
    colors = [OKABE[2]] + [OKABE[5]] * (len(rows) - 1)
    meds = [r[1] for r in rows]
    err = [[r[1] - r[2][0] for r in rows], [r[2][1] - r[1] for r in rows]]
    axes[0].bar(x, meds, yerr=err, color=colors, capsize=3)
    axes[0].set_ylim(min(meds) * 0.85, max(r[2][1] for r in rows) * 1.05)
    axes[0].set_ylabel("STT final decode, median ms (95% CI)")
    axes[0].set_title("(a) the occupancy tax", fontsize=9)
    axes[1].bar(x, [r[4] for r in rows], color=colors)
    axes[1].set_ylabel("AnonHugePages after decode (MB)")
    axes[1].set_title("(b) huge pages lost", fontsize=9)
    axes[2].bar(x, [r[3] for r in rows], color=colors)
    axes[2].set_ylabel("thp_fault_fallback per decode")
    axes[2].set_title("(c) THP fault fallback", fontsize=9)
    for ax in axes:
        ax.set_xticks(x, [r[0] for r in rows], rotation=40, ha="right", fontsize=7)
    for i, r in enumerate(rows):
        axes[0].text(i, r[2][1], f"n={r[5]}", ha="center", va="bottom", fontsize=6)
    fig.suptitle(
        "Fig 2. Residency alone taxes the recognizer: a co-resident footprint "
        "costs it its huge pages",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out, "paper_fig2_occupancy_tax")


# ---------------------------------------------------------------- Fig 3 ----
def fig3(out: Path) -> None:
    op = _script("overlap_penalties")
    import csv

    with open(REPO / "results/raw/audio/sixteen/manifest.csv", encoding="utf-8") as fh:
        refs = [r["transcript"] for r in csv.DictReader(fh)]
    th = op.thinker(RAW, op.POLICY_RUNS, refs)
    can = op.per_item_ttfa(RAW, op.CANONICAL_DEV, refs)
    nop = op.per_item_ttfa(RAW, op.NOPARTIAL_DEV, refs)
    interv = [can[i] - nop[i] for i in sorted(set(can) & set(nop))]
    # Listener c is the registered P2a fit (p2_score.py headline coding), so the
    # figure and the ledger carry one number.
    p2 = _script("p2_score")
    pop = p2.population(p2.collect(p2.runs()))
    beta, c_lo, c_hi, _kept = p2.fit_c(pop, key=p2.merged)
    c = float(beta[2])
    rows = [r for _, _, r in pop]
    med_ov = statistics.median(r["overlap_s"] for r in rows if r["overlap_s"] > 0)
    bars = [
        (
            "thinker: draft finished\nbefore the endpoint",
            statistics.median(th["finished"]),
            boot(th["finished"]),
            len(th["finished"]),
            "TTFA",
            OKABE[1],
        ),
        (
            "thinker: draft still\nrunning at the endpoint",
            statistics.median(th["still running"]),
            boot(th["still running"]),
            len(th["still running"]),
            "TTFA",
            OKABE[4],
        ),
        (
            f"listener: partial in flight\n(model: c x {med_ov:.2f} s median overlap)",
            c * med_ov,
            (c_lo * med_ov, c_hi * med_ov),
            sum(1 for r in rows if r["overlap_s"] > 0),
            "final decode",
            OKABE[0],
        ),
        (
            "listener: partial decodes\nremoved (pass 1n, dev)",
            statistics.median(interv),
            boot(interv),
            len(interv),
            "TTFA",
            OKABE[5],
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = list(range(len(bars)))
    ax.bar(
        x,
        [b_[1] for b_ in bars],
        yerr=[[b_[1] - b_[2][0] for b_ in bars], [b_[2][1] - b_[1] for b_ in bars]],
        color=[b_[5] for b_ in bars],
        capsize=3,
    )
    ax.axhline(0, color="black", lw=0.6)
    for i, b_ in enumerate(bars):
        ax.text(i, b_[2][1] + 15, f"{b_[1]:+.0f} ms\n({b_[4]})\nn={b_[3]}", ha="center", fontsize=7)
    ax.set_xticks(x, [b_[0] for b_ in bars], fontsize=7)
    ax.set_ylabel("added latency (ms) with 95% CI")
    ax.set_ylim(min(0, min(b_[2][0] for b_ in bars)) - 50, max(b_[2][1] for b_ in bars) * 1.35)
    ax.set_title(
        f"Fig 3. Two consumers, two overlap penalties (listener c = {c:.0f} ms per s of overlap)",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out, "paper_fig3_overlap_penalties")


# ---------------------------------------------------------------- Fig 4 ----
# Pass 8 (NOTES "PASS 8 (f*) ..." and its CORRECTION): reactive_mqa.yaml, two reps.
FIG4_RUNS = ("reactive-20261009-105401-6ac01d", "reactive-20261009-112837-d4ef5e")
FSTAR_PREDICTED = 0.858  # REGISTERED 2026-09-22d, a prediction, not a measurement


def fig4(out: Path) -> None:
    fs = _script("fstar_score")
    pts: list[tuple[float, float]] = []
    for run in FIG4_RUNS:
        lines = [
            json.loads(x) for x in (RAW / run / "turns.jsonl").read_text().splitlines() if x.strip()
        ]
        good = fs.scored_turns(lines)
        for p in (r for r in lines if r["kind"] == "partial_record" and r["turn"] in good):
            d = good[p["turn"]]["stages_ms"].get("speech_end_est")
            if not d or p.get("decode_done_ms", -1) < 0 or p.get("issued_ms", -1) < 0:
                continue
            lead = p["offset_s"] * 1000 - p["issued_ms"]
            pts.append((p["offset_s"] * 1000 / (d + lead), d - p["decode_done_ms"]))
    measured = []
    for run_f in ("buffer",):
        per: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for run in FIG4_RUNS:
            res, _ = fs.fstar_of_run(RAW / run, run_f, False)
            for u, v in res.items():
                per[u].append(v)
        items = [
            (statistics.median(x[0] for x in v), statistics.median(x[1] for x in v))
            for v in per.values()
        ]
        ds = sorted(d for _, d in items)
        q1, q3 = ds[len(ds) // 4], ds[(3 * len(ds)) // 4]
        measured = [f for f, d in items if q1 <= d <= q3]
    m = statistics.median(measured)
    mlo, mhi = boot(measured)
    bins: dict[float, list[float]] = defaultdict(list)
    for f, w in pts:
        if 0 <= f <= 1.05:
            bins[round(f * 20) / 20].append(w)
    bx = sorted(k for k, v in bins.items() if len(v) >= 5)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax2 = ax.twinx()
    fx = [p[0] for p in MULTI_STEP_GREEDY_P_USABLE]
    fy = [p[1] for p in MULTI_STEP_GREEDY_P_USABLE]
    ax.plot(fx, fy, "o-", color=OKABE[4], label="greedy p_usable (offline probe)")
    ax.axhline(BREAK_EVEN_P_USABLE, color=OKABE[4], ls=":", lw=1)
    ax.text(
        0.21,
        BREAK_EVEN_P_USABLE + 0.004,
        f"break-even p* = {BREAK_EVEN_P_USABLE:.3f}",
        color=OKABE[4],
        fontsize=7,
    )
    grid = [i / 100 for i in range(20, 101)]
    ax2.plot(
        grid,
        [window_model_ms(f, 15.4) for f in grid],
        color=OKABE[5],
        lw=1.2,
        label="window(f), registered model (D=15.4 s)",
    )
    ax2.plot(
        bx,
        [statistics.median(bins[k]) for k in bx],
        "s",
        color=OKABE[5],
        ms=4,
        label=f"window(f), measured, median per 0.05 bin (pass 8, {len(pts)} partials)",
    )
    ax2.axhline(590, color=OKABE[5], ls=":", lw=1)
    ax2.text(0.21, 640, "590 ms: one 10-token draft", color=OKABE[5], fontsize=7)
    ax.axvline(FSTAR_PREDICTED, color="black", ls="--", lw=1)
    ax.axvspan(mlo, mhi, color="grey", alpha=0.25)
    ax.axvline(m, color="black", lw=1)
    ax.text(
        FSTAR_PREDICTED - 0.008,
        0.012,
        f"f* predicted {FSTAR_PREDICTED}",
        rotation=90,
        ha="right",
        fontsize=7,
    )
    ax.text(
        m + 0.008, 0.012, f"f* measured {m:.3f} [{mlo:.3f}, {mhi:.3f}]", rotation=90, fontsize=7
    )
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0, 0.25)
    ax2.set_ylim(-1500, 11000)
    ax.set_xlabel("fraction f of the utterance heard (audio position at issue / speech length)")
    ax.set_ylabel("p_usable (draft usable as the answer)", color=OKABE[4])
    ax2.set_ylabel("window: speech left after the partial lands (ms)", color=OKABE[5])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        fontsize=6.5,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.42, 1.0),
    )
    ax.set_ylim(0, 0.30)
    fig.suptitle(
        "Fig 4. Usability rises with f, the window falls; they meet where no draft fits", fontsize=9
    )
    fig.tight_layout()
    save(fig, out, "paper_fig4_usable_vs_window")


# ---------------------------------------------------------------- Fig 5 ----
# Pass 2k (NOTES "PASS 2k: P3 HOLDS" and its CORRECTION) and pass 2 (NOTES
# "PASS 2 SCORED"), each against its own NOPARTIAL.
P2K = {
    "multi_step": (
        (
            "reactive-20261008-103300-abd218",
            "reactive-20261008-121644-d2b5ad",
            "reactive-20261008-125425-f94b44",
        ),
        (
            "reactive-20261008-110545-1f7989",
            "reactive-20261008-114351-9ae173",
            "reactive-20261008-132718-54285c",
        ),
    ),
    "dev": (
        (
            "reactive-20261008-113854-585574",
            "reactive-20261008-125153-9aab0d",
            "reactive-20261008-140024-eda306",
        ),
        (
            "reactive-20261008-114121-6c121f",
            "reactive-20261008-124928-58fcc2",
            "reactive-20261008-140252-12b836",
        ),
    ),
}
P2 = {
    "multi_step": (
        (
            "reactive-20261005-145416-dac216",
            "reactive-20261005-153526-e45863",
            "reactive-20261005-161631-1f1168",
        ),
        FIG1_SETS["multi_step (80)"],
    ),
    "dev": (
        (
            "reactive-20261005-144439-089fb9",
            "reactive-20261005-152547-ac8aac",
            "reactive-20261005-160652-523169",
        ),
        FIG1_SETS["dev (16)"],
    ),
}
SET_DIRS = {
    "multi_step": SPOKEN / "multi_step_reasoning-seed20260922",
    "dev": REPO / "results/raw/audio/sixteen",
}


def fig5(out: Path) -> None:
    p2k, p2 = _script("p2k_score"), _script("pass2_score")
    res: dict[str, dict[str, dict[str, tuple[float, float, float, int]]]] = {
        "pass 2k": {},
        "pass 2": {},
    }
    for s, (stop, nop) in P2K.items():
        refs = p2k.references(SET_DIRS[s])
        med = p2k.item_medians([p2k.read_run(RAW / r, refs) for r in stop + nop])
        out_s = {}
        for key, sign, name in (("ttfa", +1, "ttfa"), ("wer", -1, "wer"), ("j_raw", -1, "j")):
            b_, lo, hi, n, d = p2k.paired(med, key, sign)
            out_s[name] = mean_ci(d) if name == "wer" else (b_, lo, hi, n)
        res["pass 2k"][s] = out_s
    for s, (test, base) in P2.items():
        bm, _ = p2.arm_items([RAW / r for r in base], SET_DIRS[s])
        tm, _ = p2.arm_items([RAW / r for r in test], SET_DIRS[s])
        common = sorted(set(bm) & set(tm))
        out_s = {}
        for key, sign, name in (("ttfa_ms", +1, "ttfa"), ("wer", -1, "wer"), ("energy_j", -1, "j")):
            d = [sign * (bm[u][key] - tm[u][key]) for u in common if key in bm[u] and key in tm[u]]
            lo, hi = boot(d)
            out_s[name] = mean_ci(d) if name == "wer" else (statistics.median(d), lo, hi, len(d))
        res["pass 2"][s] = out_s
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    titles = {
        "ttfa": "(a) TTFA reduction vs NOPARTIAL (ms)",
        "wer": "(b) dWER vs NOPARTIAL (mean)",
        "j": "(c) energy per turn vs NOPARTIAL (J, raw)",
    }
    for ax, metric in zip(axes, ("ttfa", "wer", "j"), strict=True):
        xs, labels = [], []
        for k, (passname, color) in enumerate((("pass 2k", OKABE[2]), ("pass 2", OKABE[6]))):
            for j, s in enumerate(("multi_step", "dev")):
                b_, lo, hi, n = res[passname][s][metric]
                xpos = j * 2.6 + k
                ax.bar(
                    xpos,
                    b_,
                    yerr=[[b_ - lo], [hi - b_]],
                    color=color,
                    capsize=3,
                    label=passname if j == 0 else None,
                )
                ax.text(xpos, hi, f"n={n}", ha="center", va="bottom", fontsize=6)
                xs.append(xpos)
                labels.append(s)
        ax.axhline(0, color="black", lw=0.6)
        if metric == "ttfa":
            ax.axhline(300, color="grey", ls=":", lw=1)
            ax.text(-0.4, 315, "P3 bar 300 ms", fontsize=6, color="grey")
        ax.set_xticks([0.5, 3.1], ["multi_step", "dev"], fontsize=8)
        ax.set_title(titles[metric], fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Fig 5. Freezing the orphan and gating short turns turns a regression into a 357 ms saving",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out, "paper_fig5_pass2k")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/figures")
    ap.add_argument("--only", default="", help="comma-separated figure numbers")
    ap.add_argument("--print-only", action="store_true", help="report values, write no file")
    a = ap.parse_args()
    global PRINT_ONLY
    PRINT_ONLY = a.print_only
    a.out_dir.mkdir(parents=True, exist_ok=True)
    figs = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5}
    for k in a.only.split(",") if a.only else figs:
        print(f"Fig {k}")
        figs[k](a.out_dir)
    _ = Counter


if __name__ == "__main__":
    main()
