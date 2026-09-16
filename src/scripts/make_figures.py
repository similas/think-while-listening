"""Phase 2 figures, rebuilt from raw logs by `make figures`.

Fig. 2 — the cost of speculation by device state. TTFA against the tokens a
speculative decode actually produced, one line per device state, bootstrap 95%
CI bars, n on every point.

IMPORTANT READING NOTE, carried on the figure itself: in Phase 2 the
speculation is DISCARDED, never used to answer. This is the COST curve, so it
can only rise. The U-shape that H2 predicts needs the GAIN side — speculation
that actually hides latency — which arrives with the policies in Phase 3. A
monotone cost curve here neither confirms nor refutes H2; it supplies the
term the H2 test will subtract from.

Fig. 3 — STT commit latency and endpoint-detection delay against the same
axis. ΔWER is deliberately absent: on the file harness the injected audio is
bit-exact and the transcripts are byte-identical across every cell (verified
2026-09-16), so ΔWER is identically zero by construction and belongs to the
live-mic set, not here.

Style: vector PDF, colourblind-safe (Okabe-Ito), no chartjunk, every axis
labelled with units.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twl.metrics import median
from twl.records import read_jsonl

REPO = Path(__file__).resolve().parents[2]
# Okabe-Ito: distinguishable under the common forms of colour blindness.
COLOURS = {"cold": "#0072B2", "adversary": "#D55E00", "warm": "#009E73"}
MARKERS = {"cold": "o", "adversary": "s", "warm": "^"}
LABELS = {
    "cold": "cold board, no co-runner",
    "adversary": "bandwidth adversary (1 core)",
    "warm": "thermally soaked (>74 °C)",
}


def bootstrap_median_ci(
    xs: list[float], seed: int = 0, n: int = 2000
) -> tuple[float, float, float]:
    if not xs:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    boots = sorted(median([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n))
    return median(xs), boots[int(0.025 * n)], boots[int(0.975 * n)]


def collect(grid_files: list[Path]) -> dict[tuple[str, int], dict[str, list[float]]]:
    """Pool every grid pass by (state, budget)."""
    cells: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for gf in grid_files:
        index = json.loads(gf.read_text())
        for c in index["cells"]:
            run_dir = REPO / "results/raw/reactive" / c["run"]
            turns_log = run_dir / "turns.jsonl"
            if not turns_log.exists():
                continue
            scored_path = run_dir / "scored.json"
            scored = json.loads(scored_path.read_text())["turns"] if scored_path.exists() else []
            eps = {t["turn"]: t.get("endpoint_delay_ms") for t in scored}
            key = (c["state"], c["budget"])
            for r in read_jsonl(str(turns_log)):
                if r.get("kind") != "turn_record" or not r["valid"]:
                    continue
                st = r["stages_ms"]
                if "stt_final" not in st or "vad_user_stopped" not in st:
                    continue
                cells[key]["stt"].append(st["stt_final"] - st["vad_user_stopped"])
                cells[key]["tokens"].append(float(r.get("spec", {}).get("tokens_produced", 0)))
                if "audio_out_first" in st and "speech_end_est" in st:
                    cells[key]["ttfa"].append(st["audio_out_first"] - st["speech_end_est"])
                if eps.get(r["turn"]) is not None:
                    cells[key]["endpoint"].append(float(eps[r["turn"]]))
                if r.get("energy_j", -1) > 0:
                    cells[key]["energy"].append(float(r["energy_j"]))
    return cells


def series(cells: dict[tuple[str, int], dict[str, list[float]]], metric: str) -> dict[str, Any]:
    out: dict[str, Any] = defaultdict(lambda: {"x": [], "y": [], "lo": [], "hi": [], "n": []})
    for (state, _budget), data in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        vals = data.get(metric, [])
        if not vals:
            continue
        med, lo, hi = bootstrap_median_ci(vals)
        out[state]["x"].append(median(data["tokens"]) if data["tokens"] else 0.0)
        out[state]["y"].append(med)
        out[state]["lo"].append(med - lo)
        out[state]["hi"].append(hi - med)
        out[state]["n"].append(len(vals))
    return out


def plot_metric(ax: Any, s: dict[str, Any], ylabel: str, title: str) -> None:
    for state, d in s.items():
        ax.errorbar(
            d["x"],
            d["y"],
            yerr=[d["lo"], d["hi"]],
            marker=MARKERS.get(state, "o"),
            color=COLOURS.get(state, "#666666"),
            capsize=3,
            linewidth=1.6,
            markersize=6,
            label=LABELS.get(state, state),
        )
        for x, y, n in zip(d["x"], d["y"], d["n"], strict=True):
            ax.annotate(
                f"n={n}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=7,
                color="#444444",
            )
    ax.set_xlabel("speculative tokens actually decoded during the turn")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-dir", type=Path, default=REPO / "results/raw/phase2")
    p.add_argument("--out-dir", type=Path, default=REPO / "results/figures")
    args = p.parse_args()

    grids = sorted(args.grid_dir.glob("phase2-grid-*.json"))
    if not grids:
        raise SystemExit(f"no grid index files in {args.grid_dir}")
    cells = collect(grids)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Fig. 2 — the cost curve.
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plot_metric(
        ax,
        series(cells, "ttfa"),
        "time to first audio (ms)",
        "Fig. 2  Cost of concurrent speculation (speculation discarded, not used)",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig2_ttfa_vs_speculation.pdf")
    fig.savefig(args.out_dir / "fig2_ttfa_vs_speculation.png", dpi=150)
    plt.close(fig)

    # Fig. 3 — what the listener pays.
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    plot_metric(axes[0], series(cells, "stt"), "STT commit latency (ms)", "STT commit latency")
    ep = series(cells, "endpoint")
    if any(d["y"] for d in ep.values()):
        plot_metric(axes[1], ep, "endpoint-detection delay (ms)", "Endpoint-detection delay")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Fig. 3  What the listener pays. ΔWER omitted: file-harness audio is bit-exact, "
        "so transcripts are identical across cells.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig3_listener_cost.pdf")
    fig.savefig(args.out_dir / "fig3_listener_cost.png", dpi=150)
    plt.close(fig)

    total = sum(len(d.get("stt", [])) for d in cells.values())
    print(f"pooled {len(grids)} grid pass(es), {len(cells)} cells, {total} valid turns")
    for (state, budget), d in sorted(cells.items()):
        print(
            f"  {state:>10} B={budget:<4} n={len(d.get('stt', [])):>3} "
            f"tokens={median(d['tokens']) if d['tokens'] else 0:>5.0f}"
        )
    print(f"wrote {args.out_dir}/fig2_ttfa_vs_speculation.pdf and fig3_listener_cost.pdf")


if __name__ == "__main__":
    main()
