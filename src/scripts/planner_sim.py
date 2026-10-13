"""Level-2 planner: a simulator calibrated from the logs, a finite-horizon DP over
elapsed time with the endpoint hazard, and the comparison the 2026-10-13 plan
asks for -- DP vs STOP+gate vs unconditional vs REACTIVE(-NOPARTIAL) -- on
multi_step. No measured runs; existing logs only.

THE QUESTION. Does a planner that reasons about when the user will stop predict
>= 100 ms of median TTFA over the feasibility gate on multi_step? That decides
whether one pre-registered live pass of STOP+DP is worth running.

EVERYTHING RUNS ON THE BUFFER CLOCK, as the harness does: the issuer's elapsed
time is the audio VAD has passed through (stt.py), the endpoint E is the buffer
length when VAD stops, and the final decodes buffer[committed_end:E]. Only the
final differs between the listener arms, so a TTFA difference is a difference
of finals.

REALITY MODEL, per item, from the logs:
  E          the buffer length at the endpoint (REACTIVE-NOPARTIAL's final audio)
  boundaries the commit points the item ever produced (segment granularity),
             over pass 2k STOP, pass 6 and pass 2 COMMIT-WL runs
  commit     a segment boundary b is committed by hypothesis k when
             b <= x_{k-1} + EPS and b <= x_k - DELTA (x = absolute buffer end);
             EPS, DELTA fitted by replaying every logged hypothesis sequence
  decode     partial decode_ms = alpha + beta * uncommitted buffer, + a residual
             drawn from the logged residuals
  final      final_ms = a + b_w * audio, b_w the within-item slope (item fixed
             effects) over pass 2k's STOP and NOPARTIAL turns, + item effect
  overlap    unconditional (no STOP) pays c * overlap, c from P2a (492 ms/s)
The DP plans with LESS than this: the duration hazard (smoothed over items) and
a population commit model cmt' = max(cmt, x_prev - L), L fitted on the logs. It
never sees an item's boundaries or E. The CLAIRVOYANT bound knows both, and
bounds every planner in this policy space from above.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np

from twl.pacing import SelfPacedIssuer

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "results/raw/reactive"
MULTI = REPO / "results/raw/spoken_mqa/multi_step_reasoning-seed20260922"
STOP_RUNS = ("20261008-103300-abd218", "20261008-121644-d2b5ad", "20261008-125425-f94b44")
NOP_RUNS = ("20261008-110545-1f7989", "20261008-114351-9ae173", "20261008-132718-54285c")
PASS6_RUNS = ("20261009-120309-a4f48d",)
PASS2_RUNS = ("20261005-145416-dac216", "20261005-153526-e45863", "20261005-161631-1f1168")
PRIOR_S = 15.399  # pass 2k / pass 6 run_meta: median wav duration of the set
DUTY = 0.6
MIN_UNCOMMITTED_S = 1.0
C_OVERLAP = 0.492  # P2a, ms of final per ms of overlap
# Pass 2's COMMIT-WL (thread worker, no STOP): its final ran 1.31x the solo
# model of its own tail on multi_step (NOTES "PASS 2 SCORED", P6, n=211). The
# unconditional arm is that design, so its final carries the measured ratio;
# the overlap it suffered is inside the ratio and is not charged again.
UNCOND_FINAL_X = 1.31
DT = 0.05  # simulation tick, s
GRID = 0.2  # DP grid, s


def _p2s():
    spec = importlib.util.spec_from_file_location("p2s", Path(__file__).with_name("pass2_score.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lines(run: str) -> list[dict]:
    return [json.loads(x) for x in (RAW / f"reactive-{run}/turns.jsonl").read_text().splitlines()]


def item_of_turns(run: str, L: list[dict], p2s, refs) -> dict[int, str]:
    tl = json.loads((RAW / f"reactive-{run}/playback_timeline.json").read_text())
    order = [Path(e[0]).name for e in tl]
    turns = [x for x in L if x["kind"] == "turn_record"]
    if all(t.get("utterance", -1) > 0 for t in turns):
        m = {t["turn"]: order[t["utterance"] - 1] for t in turns}
    else:
        m = p2s.align(turns, order, refs)
    seen: dict[str, int] = defaultdict(int)
    for f in m.values():
        seen[f] += 1
    return {t: f for t, f in m.items() if seen[f] == 1}  # a split file is no item


def scored(t: dict) -> bool:
    st = t.get("stages_ms") or {}
    return bool(t.get("valid")) and "stt_final" in st and "vad_user_stopped" in st


# ------------------------------------------------------------- calibration --
@dataclass
class Calib:
    E: dict[str, float]
    bounds: dict[str, list[float]]
    alpha: float
    beta: float
    resid: list[float]
    a: float
    b_w: float
    item_fx: dict[str, float]
    eps: float = 0.0
    delta: float = 0.0
    L: float = 0.0
    seqs: list[tuple[str, list[tuple[float, float]]]] = field(default_factory=list)
    allseqs: dict[str, list[list[tuple[float, float]]]] = field(default_factory=dict)
    p_b: dict[str, list[float]] = field(default_factory=dict)
    measured_final: dict[str, dict[str, float]] = field(default_factory=dict)


def calibrate() -> Calib:
    p2s = _p2s()
    refs = dict(p2s.load_refs(MULTI))
    E: dict[str, list[float]] = defaultdict(list)
    fin: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    bset: dict[str, list[float]] = defaultdict(list)
    dec: list[tuple[float, float]] = []
    seqs: list[tuple[str, list[tuple[float, float]]]] = []
    allseqs: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    for arm, runs in (
        ("NOP", NOP_RUNS),
        ("STOP", STOP_RUNS),
        ("P6", PASS6_RUNS),
        ("P2", PASS2_RUNS),
    ):
        for run in runs:
            L = lines(run)
            items = item_of_turns(run, L, p2s, refs)
            hyps: dict[int, list[dict]] = defaultdict(list)
            for x in L:
                if x["kind"] == "hypothesis_record":
                    hyps[x["turn"]].append(x)
            for t in (x for x in L if x["kind"] == "turn_record"):
                f = items.get(t["turn"])
                if f is None or not scored(t):
                    continue
                st = t["stages_ms"]
                fms = st["stt_final"] - st["vad_user_stopped"]
                if arm == "NOP":
                    E[f].append(t["stt_audio_s"])
                if arm in ("NOP", "STOP"):
                    fin[f][arm].append((t["stt_audio_s"], fms))
                cmt, seq = 0.0, []
                for h in hyps.get(t["turn"], []):
                    x = cmt + h["buffer_s"]
                    if arm in ("STOP", "P6"):
                        dec.append((h["buffer_s"], h["decode_ms"]))
                    seq.append((x, h["committed_end_s"]))
                    if h["committed_end_s"] > cmt + 1e-6:
                        bset[f].append(h["committed_end_s"])
                    cmt = h["committed_end_s"]
                if arm in ("STOP", "P6") and seq:
                    seqs.append((f, seq))
                if arm != "NOP":
                    allseqs[f].append(seq)
    Em = {f: statistics.median(v) for f, v in E.items()}
    bounds = {f: _cluster(bset.get(f, [])) for f in set(Em) | set(bset) | {f for f, _ in seqs}}
    d = np.asarray(dec)
    X = np.c_[np.ones(len(d)), d[:, 0]]
    alpha, beta = np.linalg.lstsq(X, d[:, 1], rcond=None)[0]
    resid = list(d[:, 1] - X @ np.array([alpha, beta]))
    # final: within-item slope over STOP and NOP turns (item fixed effects)
    ys, xs, fx = [], [], {}
    for arms in fin.values():
        rows = [r for a_ in ("NOP", "STOP") for r in arms.get(a_, [])]
        if len(rows) < 2:
            continue
        mx = statistics.fmean(r[0] for r in rows)
        my = statistics.fmean(r[1] for r in rows)
        xs += [r[0] - mx for r in rows]
        ys += [r[1] - my for r in rows]
    b_w = float(np.dot(xs, ys) / np.dot(xs, xs))
    allrows = [(f, r) for f, arms in fin.items() for a_ in arms for r in arms[a_]]
    a = statistics.median(r[1] - b_w * r[0] for _, r in allrows)
    for f, arms in fin.items():
        rows = [r for a_ in arms for r in arms[a_]]
        fx[f] = statistics.median(r[1] - b_w * r[0] for r in rows) - a
    measured = {
        f: {a_: statistics.median(r[1] for r in arms[a_]) for a_ in arms} for f, arms in fin.items()
    }
    cal = Calib(Em, bounds, float(alpha), float(beta), resid, a, b_w, fx, seqs=seqs)
    cal.measured_final = measured
    cal.allseqs = dict(allseqs)
    fit_commit_rule(cal)
    fit_boundary_rates(cal)
    return cal


def fit_boundary_rates(cal: Calib) -> None:
    """Whether a boundary commits is not certain: it needs two hypotheses whose
    segmentation agrees there. p_b = runs that committed b / runs whose
    hypotheses satisfied the commit rule's reach for b (pass 2k, pass 6, pass 2)."""
    rates = []
    for f, B in cal.bounds.items():
        ps = []
        for b in B:
            elig = hit = 0
            for seq in cal.allseqs.get(f, []):
                reach = any(
                    seq[k - 1][0] + cal.eps >= b and seq[k][0] - cal.delta >= b
                    for k in range(1, len(seq))
                )
                done = any(abs(c - b) < 0.16 for _, c in seq)
                elig += reach or done
                hit += done
            ps.append(hit / elig if elig else 0.0)
        cal.p_b[f] = ps
        rates += ps
    print(
        f"boundary commit rate p_b: median {statistics.median(rates):.2f} "
        f"over {len(rates)} boundaries"
    )


def fill_censored(cal: Calib, p_fill: float | None = None) -> None:
    """SENSITIVITY. Boundaries no logged run reached are invisible, and the gate
    stops early on long items -- so the simulator could understate what a later
    commit would gain. Add synthetic boundaries after each item's last observed
    one, spaced by the median observed gap between consecutive boundaries, up
    to E - 0.5 s, each committable at the median p_b."""
    gaps = [b2 - b1 for B in cal.bounds.values() for b1, b2 in pairwise(B)]
    g = statistics.median(gaps)
    p = statistics.median(x for v in cal.p_b.values() for x in v) if p_fill is None else p_fill
    added = 0
    for f, B in cal.bounds.items():
        last = max(B, default=0.0)
        while f in cal.E and last + g <= cal.E[f] - 0.5:
            last += g
            B.append(last)
            cal.p_b[f].append(p)
            added += 1
    print(f"[sensitivity] filled {added} synthetic boundaries, gap {g:.2f} s, p_b {p:.2f}")


def active(cal: Calib, f: str, rng: random.Random) -> list[float]:
    return [b for b, p in zip(cal.bounds[f], cal.p_b[f], strict=True) if rng.random() < p]


def _cluster(v: list[float], tol: float = 0.15) -> list[float]:
    out: list[float] = []
    for b in sorted(v):
        if not out or b - out[-1] > tol:
            out.append(b)
    return out


def commit_after(
    bounds: list[float], cmt: float, x_prev: float, x_cur: float, eps: float, delta: float
) -> float:
    ok = [b for b in bounds if cmt + 1e-6 < b <= x_prev + eps and b <= x_cur - delta]
    return max(ok) if ok else cmt


def fit_commit_rule(cal: Calib) -> None:
    best = (-1.0, 0.0, 0.0)
    for eps in np.arange(-0.4, 1.21, 0.1):
        for delta in np.arange(0.0, 2.41, 0.1):
            hit = n = 0
            for f, seq in cal.seqs:
                cmt = 0.0
                for k in range(1, len(seq)):
                    pred = commit_after(cal.bounds[f], cmt, seq[k - 1][0], seq[k][0], eps, delta)
                    hit += abs(pred - seq[k][1]) < 0.16
                    n += 1
                    cmt = seq[k][1]
            if hit / n > best[0]:
                best = (hit / n, float(eps), float(delta))
    cal.eps, cal.delta = best[1], best[2]
    gaps = [seq[k - 1][0] - seq[k][1] for _, seq in cal.seqs for k in range(1, len(seq))]
    cal.L = statistics.median(gaps)
    print(
        f"commit rule: EPS {cal.eps:+.1f} s, DELTA {cal.delta:.1f} s, replay accuracy "
        f"{best[0]:.3f} over {sum(len(s) - 1 for _, s in cal.seqs)} hypotheses; "
        f"DP population lag L = median(x_prev - committed after) = {cal.L:.2f} s"
    )


# ---------------------------------------------------------------- simulator --
@dataclass
class Outcome:
    cmt: float
    overlap_s: float
    hyps: int


def simulate(
    cal: Calib, f: str, policy, rng: random.Random, B: list[float], stop: bool = True
) -> Outcome:
    """One turn of item f under policy(t, cmt, x_prev, x_pp, idle_s, n_hyp) -> bool."""
    E = cal.E[f]
    t, cmt, n = 0.0, 0.0, 0
    x_prev = x_pp = -1.0
    busy_until, last_end, x_issue = -1.0, 0.0, 0.0
    while t < E:
        if busy_until >= 0 and t >= busy_until:
            x_new = x_issue
            if x_prev >= 0:
                cmt = commit_after(B, cmt, x_prev, x_new, cal.eps, cal.delta)
            x_pp, x_prev = x_prev, x_new
            busy_until, last_end = -1.0, t
        if busy_until < 0 and policy(t, cmt, x_prev, x_pp, t - last_end, n):
            d = (cal.alpha + cal.beta * (t - cmt) + rng.choice(cal.resid)) / 1000.0
            busy_until, x_issue, n = t + max(0.2, d), t, n + 1
            # The harness feeds the issuer the MEASURED decode (stt.py note_decode);
            # the simulator feeds it the simulated one.
            note = getattr(policy, "note", None)
            if note is not None:
                note(max(0.2, d) * 1000.0)
        t += DT
    overlap = 0.0 if (stop or busy_until < 0) else busy_until - E
    return Outcome(cmt, overlap, n)


def final_ms(cal: Calib, f: str, o: Outcome) -> float:
    return (
        cal.a
        + cal.item_fx.get(f, 0.0)
        + cal.b_w * (cal.E[f] - o.cmt)
        + C_OVERLAP * 1000 * o.overlap_s
    )


def gate_policy() -> object:
    iss = SelfPacedIssuer(
        duty_max=DUTY, min_uncommitted_s=MIN_UNCOMMITTED_S, duration_prior_s=PRIOR_S
    )

    def pol(t, cmt, x_prev, x_pp, idle_s, n):
        if n == 0:
            iss.reset()
        dec = iss.decide(
            in_flight=False,
            uncommitted_s=t - cmt,
            idle_ms=idle_s * 1000 if n else 1e9,
            buffer_s=t - cmt,
            elapsed_s=t,
            pending_agreement=min(1, n),
        )
        return dec.issue

    pol.note = iss.note_decode  # type: ignore[attr-defined]
    return pol


def uncond_policy(cal: Calib) -> object:
    iss = SelfPacedIssuer(duty_max=DUTY, min_uncommitted_s=MIN_UNCOMMITTED_S, duration_prior_s=0.0)

    def pol(t, cmt, x_prev, x_pp, idle_s, n):
        if n == 0:
            iss.reset()
        dec = iss.decide(
            in_flight=False,
            uncommitted_s=t - cmt,
            idle_ms=idle_s * 1000 if n else 1e9,
            buffer_s=t - cmt,
            elapsed_s=t,
        )
        return dec.issue

    pol.note = iss.note_decode  # type: ignore[attr-defined]
    return pol


# ---------------------------------------------------------------------- DP --
class DP:
    """Finite-horizon DP over buffer time t on a GRID, state (t, x_prev, x_pp).

    The planner knows the endpoint hazard (smoothed duration distribution) and a
    population commit model, cmt = max(0, x_pp - L) once two hypotheses exist and
    cmt' = x_prev - L after the next completes. Objective: maximise the expected
    committed audio at the endpoint, which minimises the expected final (STOP
    removes the overlap term). Actions at an idle tick: issue now, or wait one
    GRID step. duty: if set, the same rest rule the gate obeys.
    """

    def __init__(self, cal: Calib, Es: list[float], duty: float | None) -> None:
        self.cal, self.duty = cal, duty
        self.T = int(max(Es) / GRID) + 40
        grid = np.arange(self.T + 1) * GRID
        dens = np.zeros(self.T + 1)
        for e in Es:  # Gaussian-smoothed duration density, 0.75 s bandwidth
            dens += np.exp(-0.5 * ((grid - e) / 0.75) ** 2)
        dens /= dens.sum()
        self.surv = np.concatenate([dens[::-1].cumsum()[::-1], [0.0]])  # P(E >= t)
        self.L = cal.L
        self.solve()

    def dur_steps(self, t_i: int, cmt: float) -> int:
        d = (self.cal.alpha + self.cal.beta * max(0.0, t_i * GRID - cmt)) / 1000.0
        return max(1, round(d / GRID))

    def cmt_of(self, xpp: int) -> float:
        return 0.0 if xpp < 0 else max(0.0, xpp * GRID - self.L)

    def solve(self) -> None:
        T = self.T
        # V[t][x_prev+1, x_pp+1]: expected final cmt, given idle at t, endpoint not yet reached.
        # Index 0 stands for "none". The rest rule needs the last decode's length: under the
        # duty constraint the planner must wait (1/duty - 1) * d after a completion, which
        # we fold in by making a completion land at t' + rest.
        self.V = np.zeros((T + 2, T + 2, T + 2), dtype=np.float32) if T < 300 else None
        if self.V is None:
            raise ValueError("horizon too long for the dense DP")
        self.act = np.zeros_like(self.V, dtype=bool)
        V, S = self.V, self.surv
        for t in range(T, -1, -1):
            for xp in range(-1, t):
                for xpp in range(-1, xp if xp >= 0 else 0):
                    if xp < 0 and xpp >= 0:
                        continue
                    stay = self.cmt_of(xpp)
                    # wait one step
                    p_cont = S[t + 1] / S[t] if S[t] > 0 else 0.0
                    wait = (1 - p_cont) * stay + p_cont * (
                        V[t + 1, xp + 1, xpp + 1] if t + 1 <= T else stay
                    )
                    # issue now
                    d = self.dur_steps(t, self.cmt_of(xpp))
                    rest = round((1 / self.duty - 1) * d) if self.duty else 0
                    tc = t + d
                    if tc + rest > T:
                        issue = stay
                    else:
                        p_done = S[tc] / S[t] if S[t] > 0 else 0.0
                        new_xpp = xp
                        new_cmt = self.cmt_of(new_xpp)
                        # after completion the planner idles `rest` steps, during which the
                        # endpoint may arrive: the commit made at completion counts either way
                        p_after = S[tc + rest] / S[tc] if S[tc] > 0 else 0.0
                        cont = V[tc + rest, t + 1, new_xpp + 1]
                        issue = (1 - p_done) * stay + p_done * (
                            (1 - p_after) * new_cmt + p_after * cont
                        )
                    V[t, xp + 1, xpp + 1] = max(wait, issue)
                    self.act[t, xp + 1, xpp + 1] = issue >= wait

    def policy(self) -> object:
        def pol(t, cmt, x_prev, x_pp, idle_s, n):
            ti = min(self.T, round(t / GRID))
            xp = -1 if x_prev < 0 else min(ti - 1, round(x_prev / GRID))
            xpp = -1 if x_pp < 0 else min(max(xp - 1, -1), round(x_pp / GRID))
            if self.duty and n:
                d_last = self.dur_steps(ti, cmt) * GRID
                if idle_s < (1 / self.duty - 1) * d_last:
                    return False
            return bool(self.act[ti, xp + 1, xpp + 1])

        return pol

    def table(self, path: Path) -> None:
        """For each x_prev (as time since it) and t: does the DP issue? On-path states
        only: x_pp one decode before x_prev."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("t_s,since_last_hyp_s,issue\n")
            for t in range(0, self.T + 1, 5):
                for back in range(0, min(t, 60) + 1, 5):
                    xp = t - back
                    xpp = max(-1, xp - 15)
                    fh.write(
                        f"{t * GRID:.1f},{back * GRID:.1f},{int(self.act[t, xp + 1, xpp + 1])}\n"
                    )


# ------------------------------------------------------------------- report --
def clairvoyant_cmt(cal: Calib, f: str, B: list[float]) -> float:
    """Best commit any planner could reach knowing E and which boundaries will
    commit this time (mean decode)."""
    E, best = cal.E[f], 0.0
    for b in B:
        x1 = b - cal.eps
        d1 = (cal.alpha + cal.beta * x1) / 1000.0
        x2 = max(b + cal.delta, x1 + d1)
        d2 = (cal.alpha + cal.beta * max(0.0, x2 - 0.0)) / 1000.0
        if x1 >= 0 and x2 + d2 <= E:
            best = max(best, b)
    return best


def boot_median(v: list[float], reps: int = 4000) -> tuple[float, float]:
    rng = random.Random(0)
    m = sorted(statistics.median(v[rng.randrange(len(v))] for _ in v) for _ in range(reps))
    return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--fill", action="store_true", help="sensitivity: fill censored boundaries")
    ap.add_argument("--fill-p", type=float, default=None, help="commit rate of filled boundaries")
    ap.add_argument("--table", type=Path, default=REPO / "results/tables/planner_policy_table.csv")
    a = ap.parse_args()
    cal = calibrate()
    if a.fill:
        fill_censored(cal, a.fill_p)
    items = sorted(f for f in cal.E if f in cal.item_fx)
    cover = [max(cal.bounds[f], default=0.0) / cal.E[f] for f in items]
    print(
        f"items {len(items)}; E median {statistics.median(cal.E[f] for f in items):.2f} s; "
        f"partial decode {cal.alpha:.0f} + {cal.beta:.1f} ms/s; final {cal.a:.0f} + "
        f"{cal.b_w:.1f} ms/s (within-item); boundaries per item median "
        f"{statistics.median(len(cal.bounds[f]) for f in items)}, last boundary at median "
        f"{statistics.median(cover):.2f} of E"
    )
    Es = [cal.E[f] for f in items]
    dps = {"DP (duty 0.6)": DP(cal, Es, DUTY), "DP (no duty bound)": DP(cal, Es, None)}
    dps["DP (duty 0.6)"].table(a.table)
    print(f"policy table: {a.table}")
    arms = {
        "REACTIVE-NOPARTIAL": None,
        "unconditional (no gate, no STOP)": "uncond",
        "STOP + gate": "gate",
        **{k: k for k in dps},
    }
    res: dict[str, dict[str, float]] = defaultdict(dict)
    hyps: dict[str, list[int]] = defaultdict(list)
    frac: dict[str, list[float]] = defaultdict(list)
    for f in items:
        # COMMON RANDOM NUMBERS: one draw of which boundaries commit, per item and
        # replicate, shared by every arm, so a paired difference is the policy's.
        crn = [
            active(cal, f, random.Random(zlib.crc32(f"{f}|{r}|b".encode()))) for r in range(a.reps)
        ]
        for arm, kind in arms.items():
            if kind is None:
                res[arm][f] = final_ms(cal, f, Outcome(0.0, 0.0, 0))
                continue
            vals = []
            for r in range(a.reps):
                rng = random.Random(zlib.crc32(f"{f}|{r}|d".encode()))
                if kind == "gate":
                    o = simulate(cal, f, gate_policy(), rng, crn[r])
                elif kind == "uncond":
                    o = simulate(cal, f, uncond_policy(cal), rng, crn[r])
                    vals.append(UNCOND_FINAL_X * final_ms(cal, f, o))
                    hyps[arm].append(o.hyps)
                    frac[arm].append(o.cmt / cal.E[f])
                    continue
                else:
                    o = simulate(cal, f, dps[kind].policy(), rng, crn[r])
                vals.append(final_ms(cal, f, o))
                hyps[arm].append(o.hyps)
                frac[arm].append(o.cmt / cal.E[f])
            res[arm][f] = statistics.fmean(vals)
        res["clairvoyant bound"][f] = statistics.fmean(
            final_ms(cal, f, Outcome(clairvoyant_cmt(cal, f, crn[r]), 0.0, 0))
            for r in range(a.reps)
        )

    def paired(base: str, test: str) -> str:
        d = [res[base][f] - res[test][f] for f in items]
        lo, hi = boot_median(d)
        return f"{statistics.median(d):+.0f} ms [{lo:+.0f}, {hi:+.0f}] n={len(d)}"

    print("\nVALIDATION against pass 2k (measured +357 ms [+294, +412] n=70):")
    print(
        "  simulated TTFA reduction, NOPARTIAL - STOP+gate: "
        + paired("REACTIVE-NOPARTIAL", "STOP + gate")
    )
    meas = [
        cal.measured_final[f]["NOP"] - cal.measured_final[f]["STOP"]
        for f in items
        if {"NOP", "STOP"} <= set(cal.measured_final.get(f, {}))
    ]
    print(
        f"  measured on these items (final only): {statistics.median(meas):+.0f} ms n={len(meas)}"
    )
    print(
        "  [report] unconditional vs NOPARTIAL: "
        + paired("REACTIVE-NOPARTIAL", "unconditional (no gate, no STOP)")
        + " (pass 2 measured -155 [-487, +59], older build, thread worker)"
    )
    print("\nCOMPARISON, median paired final-decode (= TTFA) reduction vs STOP + gate, multi_step:")
    for arm in ("DP (duty 0.6)", "DP (no duty bound)", "clairvoyant bound"):
        d = [res["STOP + gate"][f] - res[arm][f] for f in items]
        print(
            f"  {arm:22s} {paired('STOP + gate', arm)}; "
            f"[report] mean {statistics.fmean(d):+.0f} ms, "
            f"Q3 {statistics.quantiles(d, n=4)[2]:+.0f} ms, items gaining >= 100 ms: "
            f"{sum(x >= 100 for x in d)}/{len(d)}"
        )
    print(
        "\nCHECK ON THE BOUNDARY MODEL, committed fraction of the utterance at the endpoint "
        "(median over turns; measured: STOP + gate 0.40-0.43 per run, pass 2k; "
        "unconditional 0.42-0.46 per run, pass 2):"
    )
    for arm in ("STOP + gate", "unconditional (no gate, no STOP)"):
        print(f"  simulated {arm}: {statistics.median(frac[arm]):.2f}")
    for arm in arms:
        if hyps.get(arm):
            print(f"  [report] hypotheses per turn, {arm}: median {statistics.median(hyps[arm])}")
    gain = statistics.median(res["STOP + gate"][f] - res["DP (duty 0.6)"][f] for f in items)
    print(
        f"\nDECISION INPUT: DP (duty 0.6) over STOP+gate {gain:+.0f} ms median -> "
        + (
            ">= 100 ms: a live pass is worth considering"
            if gain >= 100
            else "< 100 ms: no live pass"
        )
    )


if __name__ == "__main__":
    main()
