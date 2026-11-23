"""Per pass: the pre-roll audit's flagged turns under each scorer's OWN filter (Ali, 2026-11-20).

Reads every turn's class from preroll_audit.py's --out file and each
registered scorer's population from src/scripts/preroll_pop/ (one module per
scorer that reads live turn records; each calls the scorer's own functions or
copies an inline filter verbatim, and reproduces the counts the scorer
prints: run a module as __main__, or this script with --verify).

Per population key it prints the turns in the population by class: stale
(the content check), unverifiable (the audit could not clear it), unchecked
(no timeline, played file or saved segment), fresh, and names every stale and
unverifiable turn. A key with "::", "__" or a dotted suffix is a GATE or a
breakdown: turns whose fields decide membership or pairing (an item mapped by
its transcript) without their values entering the number. EXPOSED lists the
claim keys, gates excluded, holding a stale or unverifiable turn;
preroll_sensitivity.py re-scores exactly those.

GATE ALIGNMENT: where a gate maps turns to items by their transcripts
(pass2_score.align, used by pass 2, P2b and the planner), each flagged turn's
item is compared with the file its located buffer spans in the delivered
stream; a turn the audit could not locate is named.

Scorers that read no live turn record (worker_memory.py, score_prefix_probe.py)
have no module.

    preroll_exposure.py --audit FILE.jsonl [--verify] [--log FILE]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import pkgutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import scripts.preroll_pop as pops

FLAGGED = ("stale", "unverifiable")


def is_gate(key: str) -> bool:
    return "::" in key or "__" in key or "." in key


def classes(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[(str(r["run"]), int(r["turn"]))] = r
    return out


def describe(r: dict[str, object]) -> str:
    tag = f"{str(r['run'])[-6:]} t{r['turn']} {r['cls']}"
    return tag + (f" {r['stale_ms']} ms" if r["cls"] == "stale" else f" ({r['why']})")


def exposure(
    cls: dict[tuple[str, int], dict[str, object]],
    flagged: tuple[str, ...] = FLAGGED,
) -> tuple[list[str], dict[str, list[dict[str, object]]], dict[str, set[tuple[str, int]]]]:
    """(the EXPOSURE lines, the flagged rows per exposed claim, the gate sets)."""
    lines: list[str] = []
    exposed: dict[str, list[dict[str, object]]] = {}
    gates: dict[str, set[tuple[str, int]]] = {}
    for mod in sorted(m.name for m in pkgutil.iter_modules(pops.__path__)):
        m = importlib.import_module(f"scripts.preroll_pop.{mod}")
        with contextlib.redirect_stdout(io.StringIO()):
            population = m.population()
        for key, turns in population.items():
            c: Counter[str] = Counter()
            why: Counter[str] = Counter()
            flagged_rows = []
            for run, turn in sorted(turns):
                r = cls.get((run, int(turn)))
                if r is None:
                    c["not in the audit"] += 1
                    continue
                c[str(r["cls"])] += 1
                if r["cls"] == "unchecked":
                    why[str(r["why"])] += 1
                if r["cls"] in flagged:
                    flagged_rows.append(r)
            line = (
                f"EXPOSURE {mod.removeprefix('pop_')} {key}{' [gate]' if is_gate(key) else ''}: "
                f"turns {len(turns)}; stale {c['stale']}; unverifiable {c['unverifiable']}; "
                f"unchecked {c['unchecked']}{' ' + str(dict(why)) if why else ''}; fresh "
                f"{c['fresh']}"
            )
            if c["not in the audit"]:
                line += f"; not in the audit {c['not in the audit']}"
            if flagged_rows:
                line += f"; flagged: {', '.join(describe(r) for r in flagged_rows)}"
            lines.append(line)
            if is_gate(key):
                gates[f"{mod}:{key}"] = {(r, int(t)) for r, t in turns}
            elif flagged_rows:
                exposed[key] = flagged_rows
    return lines, exposed, gates


def item_map(gate: str, run: str, refs: dict[str, dict[str, str]]) -> tuple[str, dict[int, str]]:
    """(how, turn -> file): the item map the gate's own scorer builds for a run."""
    import scripts.p2_score as p2
    import scripts.pass2_score as p2s
    import scripts.planner_sim as planner
    from scripts.paper_sources import REPO

    d = REPO / "results/raw/reactive" / run
    lines = [json.loads(x) for x in (d / "turns.jsonl").read_text().splitlines() if x.strip()]
    recs = [x for x in lines if x.get("kind") == "turn_record"]
    order = [Path(e[0]).name for e in json.loads((d / "playback_timeline.json").read_text())]
    if gate.startswith("pop_planner_sim:"):  # planner_sim.item_of_turns
        how = "utterance" if all(t.get("utterance", -1) > 0 for t in recs) else "transcript"
        return how, planner.item_of_turns(run.removeprefix("reactive-"), lines, p2s, refs["multi"])
    if gate.startswith("pop_p2_score:"):  # p2_score.main's items_of rule
        rows = p2.turn_rows(d)
        if rows and all(r["utterance"] > 0 for r in rows):
            return "utterance", {r["turn"]: order[r["utterance"] - 1] for r in rows}
        return "transcript", p2s.align(recs, order, refs["multi"])
    # pass2_score.read_run: align() over every turn record, multi_step or dev
    return "transcript", p2s.align(recs, order, refs["dev" if "pass2_dev" in gate else "multi"])


def gate_alignment(
    cls: dict[tuple[str, int], dict[str, object]], gates: dict[str, set[tuple[str, int]]]
) -> list[str]:
    """Per gate that maps turns to items (alignment inputs, item maps): does the
    scorer's own map put each flagged turn on the file its located buffer
    spans? Runs it maps by the recorded utterance are not transcript-dependent."""
    import scripts.pass2_score as p2s
    from scripts.paper_sources import MULTI, REPO, SOURCES

    dev = SOURCES["pass2dev"].args[list(SOURCES["pass2dev"].args).index("--set-dir") + 1]
    refs = {
        "multi": dict(p2s.load_refs(REPO / MULTI)),
        "dev": dict(p2s.load_refs(REPO / dev)),
    }
    lines = []
    mapping = [g for g in gates if "alignment_inputs" in g or "item_map" in g]
    for gate in sorted(mapping):
        flagged_turns = {
            (r, t) for r, t in gates[gate] if cls.get((r, t), {}).get("cls") in FLAGGED
        }
        if not flagged_turns:
            lines.append(f"GATE ALIGNMENT {gate}: no flagged turn")
            continue
        agree = by_utterance = 0
        disagree: list[str] = []
        unlocated: list[str] = []
        for run in sorted({r for r, _ in flagged_turns}):
            how, item = item_map(gate, run, refs)
            tl = json.loads(
                (REPO / "results/raw/reactive" / run / "playback_timeline.json").read_text()
            )
            for run_, turn in sorted(t for t in flagged_turns if t[0] == run):
                r = cls[(run_, turn)]
                if how == "utterance":
                    by_utterance += 1
                elif "close" not in r:
                    unlocated.append(describe(r))
                else:
                    lo, hi = int(str(r["open"])), int(str(r["close"]))
                    spans = {Path(p).name for p, s0, n in tl if s0 < hi and lo < s0 + n}
                    if turn not in item or item[turn] in spans:
                        agree += 1  # a turn the map drops cannot be put on the wrong item
                    else:
                        disagree.append(f"{describe(r)}: mapped to {item[turn]}")
        lines.append(
            f"GATE ALIGNMENT {gate}: flagged turns {len(flagged_turns)}; in runs mapped by "
            f"the recorded utterance {by_utterance}; by transcript, located {agree + len(disagree)}"
            f": on the file their buffer spans or unmapped {agree}, elsewhere {len(disagree)}"
            + (f" ({'; '.join(disagree)})" if disagree else "")
            + f"; not locatable {len(unlocated)}"
            + (f": {', '.join(unlocated)}" if unlocated else "")
        )
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--audit", type=Path, required=True, help="preroll_audit.py --out")
    p.add_argument("--verify", action="store_true", help="also run each module as __main__")
    p.add_argument("--log", type=Path, default=None)
    args = p.parse_args()
    build = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if args.log and build.endswith("-dirty"):
        raise SystemExit(f"build {build} is dirty: a logged run needs a clean tree")
    cls = classes(args.audit)
    lines = [f"# preroll_exposure.py build {build}; audit {args.audit}"]
    body, exposed, gates = exposure(cls)
    lines += body
    lines.append(
        f"EXPOSED claims ({len(exposed)}): "
        + "; ".join(
            f"{k} ({sum(r['cls'] == 'stale' for r in v)} stale, "
            f"{sum(r['cls'] == 'unverifiable' for r in v)} unverifiable)"
            for k, v in exposed.items()
        )
    )
    lines += gate_alignment(cls, gates)
    if args.verify:
        for mod in sorted(m.name for m in pkgutil.iter_modules(pops.__path__)):
            src = (Path(pops.__path__[0]) / f"{mod}.py").read_text()
            flag = ["--verify"] if "--verify" in src else []  # else it checks by default
            res = subprocess.run(
                [sys.executable, "-m", f"scripts.preroll_pop.{mod}", *flag],
                capture_output=True,
                text=True,
                check=False,
            )
            out = [x for x in res.stdout.splitlines() if x.strip()]
            lines.append(f"# VERIFY {mod} (exit {res.returncode}):")
            lines += [f"#   {x}" for x in out]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
