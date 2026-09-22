"""Score the offline prefix probe: gold accuracy, then p_usable(fraction, B).

Reads results/raw/spoken_mqa/prefix_probe.json and computes every number in the
pre-registration (results/NOTES.md, 2026-09-21 + amendment 2026-09-22). Nothing
here re-generates text, so a scoring change costs no GPU time.

ORDER IS PART OF THE PRE-REGISTRATION. The benchmark gate is printed first, then
the sampling control, then p_usable. The required-precision table is recomputed
from the measured p_usable, and only after that is any trigger number read.

TOKEN BUDGETS ARE EXACT. A draft at budget B is the first B tokens of the SAME
generation, cut with the server's own tokenizer (/tokenize with_pieces), not by
counting words. The pieces are written back into the raw file on first run, so
every later scoring pass is offline and identical.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import yaml

from twl.services import _CLAUSE, FIRST_CHUNK_TARGET_CHARS

BUDGETS = (16, 32, 96)
# Gate: below this the model cannot do the benchmark, and p_usable against its
# own output is self-consistency, not usable speculation.
GATE = 0.30
# From the measured arms, not from this data: results/NOTES.md.
OVERLAP_COST_MS = 100.3
SAVING_MS = 610.0
# p x saving = (1 - p) x cost, the p at which one spend pays for one mistake.
BREAK_EVEN = OVERLAP_COST_MS / (OVERLAP_COST_MS + SAVING_MS)
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
# Gemma template markers the raw endpoint sometimes emits; everything from the
# first one on is template leakage, not reply text.
LEAK = re.compile(r"<(?:start|end)_of_turn>")
# Fixed here and printed with the table, per the amendment. Deliberately small:
# it only has to catch the words a content-free opener is made of.
_STOPWORD_TEXT = (
    "a an the and or but if then so of in on at to for from with without by as is are was "
    "were be been being do does did doing have has had having i you he she it we they this "
    "that these those there here what which who whom how why when where can could should "
    "would will shall may might must not no yes let us me my your his her its our their "
    "about into over under again more most some any all each both few other than too very "
    "s t just now well okay ok sure sorry"
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def clean(text: str) -> str:
    return LEAK.split(text, maxsplit=1)[0].strip()


def first_chunk(text: str) -> str:
    """The first chunk Piper would synthesize — the one that starts speech.

    Mirrors PiperTTSService._split_ready(final=False) for the first chunk, and
    falls back to the whole reply when the reply is shorter than either rule
    (the final flush). Pinned against the service by
    src/tests/test_prefix_probe_scoring.py.
    """
    m = _CLAUSE.search(text)
    if m and m.end() >= 8:
        return text[: m.end()]
    if len(text) >= FIRST_CHUNK_TARGET_CHARS:
        cut = text.rfind(" ", 0, FIRST_CHUNK_TARGET_CHARS + 12)
        if cut > 8:
            return text[:cut]
    return text


def words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^\w\s]", " ", text.lower()).split() if w]


def is_word_prefix(draft: str, reference: str) -> bool:
    d, r = words(draft), words(reference)
    return bool(d) and len(d) <= len(r) and r[: len(d)] == d


def lcp_words(a: str, b: str) -> int:
    x, y = words(a), words(b)
    n = 0
    for u, v in zip(x, y, strict=False):
        if u != v:
            break
        n += 1
    return n


def content_free(chunk: str, problem: str) -> bool:
    """True if the matched chunk says nothing about the problem.

    No digit, and no non-stopword it shares with the problem text. "Sure, let me
    work that out." agreeing across two generations is agreement about phrasing.
    """
    if any(c.isdigit() for c in chunk):
        return False
    shared = {w for w in words(chunk) if w not in STOPWORDS} & set(words(problem))
    return not shared


def numbers(text: str) -> set[float]:
    out = set()
    for tok in NUMBER.findall(text.replace("$", "")):
        try:
            out.add(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def number_class(chunk: str, gold: list[str], problem: str) -> str:
    """What a matched chunk actually said, by the numbers in it.

    Both classes score as "usable" under the chunk-prefix rule and only one is
    worth a controller's budget: a draft that ANTICIPATES the answer, against
    one that RESTATES a premise it was just told. Gold wins ties, since a gold
    value that also appears in the problem is still the answer.
    """
    got = numbers(chunk)
    if not got:
        return "none"
    if got & numbers(" ".join(gold)):
        return "gold"
    if got & numbers(problem):
        return "premise"
    return "other"


def last_number(text: str) -> float | None:
    found = NUMBER.findall(text.replace("$", ""))
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tokenize(text: str, host: str, port: int) -> list[str]:
    body = json.dumps({"content": text, "with_pieces": True}).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/tokenize", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return [t["piece"] for t in json.loads(r.read())["tokens"]]


def enrich(data: dict, path: Path, host: str, port: int) -> None:
    """Add the server's own tokenization to every record, once."""
    missing = [r for r in data["records"] if "pieces" not in r]
    if not missing:
        return
    print(f"tokenizing {len(missing)} generations with the server's tokenizer...")
    for r in missing:
        r["pieces"] = tokenize(clean(r["text"]), host, port)
    path.write_text(json.dumps(data, indent=1) + "\n")


def score_one(
    probe: Path, items_path: Path, config_path: Path, host: str, port: int
) -> dict[str, float]:
    """Score one probe log and print its section. Returns the headline numbers.

    Returned so several logs can be compared without re-deriving anything:
    the comparison below must read the SAME quantities the tables print.
    """
    print("=" * 72)
    print(f"{probe}")

    if not probe.exists():
        # results/raw/ is local, so a fresh clone has no probe to score. Say so
        # and exit clean rather than failing `make results` for everyone.
        print(f"no probe log at {probe}; run src/scripts/prefix_probe.py first")
        return
    problems = {m["idx"]: m["transcript"] for m in json.loads(items_path.read_text())}
    cfg_offsets = yaml.safe_load(config_path.read_text())["stt"]["partial_offsets_s"]
    data = json.loads(probe.read_text())
    enrich(data, probe, host, port)
    recs = data["records"]
    by: dict[tuple[int, float, int], dict] = {(r["idx"], r["fraction"], r["twin"]): r for r in recs}
    items = sorted({r["idx"] for r in recs})
    fractions = sorted({r["fraction"] for r in recs})
    print(
        f"items {len(items)}   generations {len(recs)}   "
        f"temperature {data['temperature']}   max_tokens {data['max_tokens']}"
    )
    gen_n = sorted(r["predicted_n"] for r in recs)
    print(
        f"generation length tokens: median {gen_n[len(gen_n) // 2]}, max {gen_n[-1]} "
        f"(cap {data['max_tokens']} never binds)\n"
    )

    gate_rate = float("nan")
    print("GOLD ACCURACY — the gate. Final answer = the last number in the reply.")
    print(f"  {'fraction':>9} {'correct':>9} {'n':>4} {'Wilson 95%':>18}")
    for f in fractions:
        k = n = 0
        for i in items:
            r = by.get((i, f, 0))
            if r is None:
                continue
            n += 1
            gold = last_number(" ".join(r["gold"]))
            got = last_number(clean(r["text"]))
            k += int(gold is not None and got is not None and gold == got)
        lo, hi = wilson(k, n)
        mark = ""
        if f == 1.0:
            gate_rate = k / n
            mark = "   <- GATE" + ("  PASS" if k / n >= GATE else "  FAIL")
        print(f"  {f:>9.2f} {k / n:>9.3f} {n:>4} {f'[{lo:.3f}, {hi:.3f}]':>18}{mark}")
    print(f"  gate is {GATE:.0%} at fraction 1.00\n")

    ceiling = float("nan")
    if not data.get("twins", True):
        print("SAMPLING CONTROL: NONE, and deliberately. At temperature 0 the model")
        print("  reproduces its own output from identical input, so a twin would")
        print("  score 1.000 by construction and measure nothing. p_usable below is")
        print("  the PREFIX effect with the sampler's contribution removed, which is")
        print("  the quantity the value model needs.")
    else:
        print("SAMPLING CONTROL — two generations from the SAME full transcript.")
        print(f"  This is the ceiling: at temperature {data['temperature']} no prefix")
        print("  can do better.")
        for b in BUDGETS:
            k = n = 0
            for i in items:
                a, c = by.get((i, 1.0, 0)), by.get((i, 1.0, 1))
                if a is None or c is None:
                    continue
                n += 1
                # SAME ORIENTATION as the p_usable table below — twin 0 is the
                # draft, twin 1 the reference. is_word_prefix is not symmetric,
                # so swapping them gives a different and equally arbitrary
                # count, and the control would not line up with the 1.00 row it
                # is meant to be identical to.
                k += int(
                    is_word_prefix(
                        first_chunk("".join(a["pieces"][:b]).strip()),
                        first_chunk(clean(c["text"])),
                    )
                )
            lo, hi = wilson(k, n)
            if b == max(BUDGETS):
                ceiling = k / n if n else float("nan")
            print(f"  B={b:>3}: {k}/{n} = {k / n:.3f}  Wilson [{lo:.3f}, {hi:.3f}]")

    chunk_tokens = []
    for r in recs:
        text = clean(r["text"])
        chunk = first_chunk(text)
        n_tok = 0
        acc = ""
        for piece in r["pieces"]:
            if len(acc) >= len(chunk):
                break
            acc += piece
            n_tok += 1
        chunk_tokens.append(n_tok)
    chunk_tokens.sort()
    print(
        f"\nFIRST-CHUNK LENGTH: median {chunk_tokens[len(chunk_tokens) // 2]} tokens, "
        f"p95 {chunk_tokens[int(0.95 * len(chunk_tokens))]}, max {chunk_tokens[-1]} "
        f"(n={len(chunk_tokens)})"
    )
    print("  share of generations whose first chunk is LONGER than the budget,")
    print("  i.e. the share the budget actually cuts:")
    for b in BUDGETS:
        cut = sum(1 for t in chunk_tokens if t > b) / len(chunk_tokens)
        print(f"    B={b:>3}: {cut:.3f}")
    print("  TRUNCATION CAN ONLY HELP THE PREFIX TEST — a shorter draft chunk is")
    print("  more easily a word-prefix of the reference's — so p_usable is weakly")
    print("  DECREASING in B by construction, and a flat column means the budget")
    print("  never cut anything. That is not evidence for the flat-in-B")
    print("  assumption, which concerns whether a LONGER draft is more useful.")

    print("\np_usable — the draft's first TTS chunk is a word-prefix of the")
    print("  reference's first TTS chunk. 'content' removes matches with no digit")
    print("  and no non-stopword shared with the problem text.")
    print(f"  {'fraction':>9} " + " ".join(f"{'B=' + str(b):>20}" for b in BUDGETS))
    matched_log: list[dict] = []
    usable_rate: dict[tuple[float, int], float] = {}
    classes: dict[str, int] = {}
    for f in fractions:
        cells = []
        for b in BUDGETS:
            k = kc = n = 0
            for i in items:
                d, ref = by.get((i, f, 0)), by.get((i, 1.0, 1 if f == 1.0 else 0))
                if d is None or ref is None or d is ref:
                    continue
                n += 1
                dc = first_chunk("".join(d["pieces"][:b]).strip())
                rc = first_chunk(clean(ref["text"]))
                if is_word_prefix(dc, rc):
                    k += 1
                    cf = content_free(dc, problems.get(i, ""))
                    kc += int(not cf)
                    klass = number_class(dc, d["gold"], problems.get(i, ""))
                    classes[klass] = classes.get(klass, 0) + 1
                    matched_log.append(
                        {
                            "idx": i,
                            "fraction": f,
                            "budget": b,
                            "chunk": dc,
                            "content_free": cf,
                            "number_class": klass,
                        }
                    )
            usable_rate[(f, b)] = k / n if n else float("nan")
            cells.append(f"{k}/{n}={k / n:.2f} c{kc / n:.2f}" if n else "n/a")
        print(f"  {f:>9.2f} " + " ".join(f"{c:>20}" for c in cells))
    print("  'c' is the same cell with content-free matches removed.")
    print(f"\n  Wilson 95% on each fraction at B={max(BUDGETS)}:")
    for f in fractions:
        r = usable_rate.get((f, max(BUDGETS)))
        if r is None or r != r:
            continue
        k = round(r * len(items))
        lo, hi = wilson(k, len(items))
        print(f"  {f:>9.2f}  {k}/{len(items)} = {r:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"  stopwords ({len(STOPWORDS)}): {' '.join(sorted(STOPWORDS))}")

    out = probe.with_name(f"{probe.stem}_matches.json")
    out.write_text(json.dumps(matched_log, indent=1) + "\n")
    print(f"\n  every matched chunk logged verbatim to {out} ({len(matched_log)} matches)")

    total_matches = sum(classes.values())
    print("\nWHAT THE MATCHED CHUNKS SAY, by the numbers they contain")
    print("  (every match, every budget; gold wins ties):")
    for klass, label in (
        ("gold", "contains the GOLD number      — anticipated the answer"),
        ("premise", "contains a PREMISE number     — restated the problem"),
        ("other", "a number in neither           — arithmetic of its own"),
        ("none", "no number at all              — agreement without arithmetic"),
    ):
        k = classes.get(klass, 0)
        share = k / total_matches if total_matches else float("nan")
        print(f"  {label}: {k:>4}/{total_matches} = {share:.3f}")

    print("\nREQUIRED PRECISION recomputed at the MEASURED p_usable.")
    print("  required = cost / (cost + p_usable x saving), cost 100.3 ms")
    print("  (overlap penalty), saving 610 ms (stt_final -> tts_first_audio).")
    print(f"  {'fraction':>9} {'p_usable':>9} {'required precision':>19} {'of ceiling':>11}")
    for f in fractions:
        pu = usable_rate[(f, max(BUDGETS))]
        req = OVERLAP_COST_MS / (OVERLAP_COST_MS + pu * SAVING_MS) if pu >= 0 else float("nan")
        share = pu / ceiling if ceiling == ceiling and ceiling else float("nan")
        print(f"  {f:>9.2f} {pu:>9.3f} {req:>19.3f} {share:>11.2f}")
    if ceiling == ceiling:
        print(f"  ceiling is the sampling control, {ceiling:.3f}: two generations")
        print("  from the SAME full transcript. No prefix can beat it here.")
    else:
        print("  no ceiling column: greedy decoding has no sampling loss to bound,")
        print("  so 'of ceiling' would be p_usable against 1.000 and says nothing")
        print("  the p_usable column does not.")

    if not data.get("twins", True):
        pu = usable_rate.get((0.75, max(BUDGETS)), float("nan"))
        verdict = (
            "FALSIFIED — value side CLOSED on this corpus" if pu < BREAK_EVEN else "NOT falsified"
        )
        print("\nFALSIFICATION TEST (pre-registered 2026-09-22c), greedy arm only.")
        print(
            f"  break-even p* = {OVERLAP_COST_MS:.1f} / ({OVERLAP_COST_MS:.1f} + {SAVING_MS:.0f})"
            f" = {BREAK_EVEN:.4f}"
        )
        print(f"  greedy p_usable(0.75) = {pu:.4f}   ->   {verdict}")

    print("\nWHERE THE LIVE PIPELINE ACTUALLY DECIDES, on these utterances.")
    print("  The partial offsets are wall-clock seconds of audio; what the probe")
    print("  varies is the FRACTION of the utterance heard. Mapping one onto the")
    print("  other says which row of the table the pipeline lives in.")
    durs = sorted(r["duration_s"] for r in recs if r["fraction"] == 1.0 and r["twin"] == 0)
    med = durs[len(durs) // 2]
    for off in cfg_offsets:
        print(
            f"  partial at {off:>4.1f} s of audio -> {off / med:.2f} of the median "
            f"{med:.1f} s utterance"
        )

    print("\nSECONDARY — longest common word prefix, draft vs reference, at B=96")
    for f in fractions:
        vals = []
        for i in items:
            d, ref = by.get((i, f, 0)), by.get((i, 1.0, 1 if f == 1.0 else 0))
            if d is None or ref is None or d is ref:
                continue
            vals.append(lcp_words(clean(d["text"]), clean(ref["text"])))
        vals.sort()
        if not vals:
            # A greedy run has no twin, so fraction 1.00 has no pair to score.
            continue
        print(
            f"  {f:>9.2f}  median {vals[len(vals) // 2]:>3} words   "
            f"max {vals[-1]:>3}   n={len(vals)}"
        )

    return {
        "gate": gate_rate,
        "temperature": data["temperature"],
        "twins": float(data.get("twins", True)),
        **{f"p_usable_{f:.2f}": usable_rate[(f, max(BUDGETS))] for f in fractions},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--probe",
        type=Path,
        nargs="+",
        default=[Path("results/raw/spoken_mqa/prefix_probe.json")],
        help="one or more probe logs; several are scored and then compared",
    )
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8093)
    p.add_argument(
        "--items",
        type=Path,
        default=Path("results/raw/spoken_mqa/multi_step_reasoning.json"),
        help="the problem texts, needed for the content-free filter",
    )
    args = p.parse_args()
    args = p.parse_args()

    missing = [q for q in args.probe if not q.exists()]
    if missing:
        # results/raw/ is local, so a fresh clone has no probe to score.
        print(f"no probe log at {missing[0]}; run src/scripts/prefix_probe.py first")
        return
    summaries = {
        q.name: score_one(q, args.items, args.config, args.host, args.port) for q in args.probe
    }
    if len(summaries) < 2:
        return

    print("=" * 72)
    print("ACROSS RUNS — the same 80 items, so these are paired on the sample")
    print("  even though the generations are not paired token by token.")
    keys = sorted({k for v in summaries.values() for k in v})
    width = max(len(n) for n in summaries) + 2
    print(f"  {'quantity':>16} " + " ".join(f"{n:>{width}}" for n in summaries))
    for k in keys:
        if k in ("twins",):
            continue
        print(
            f"  {k:>16} "
            + " ".join(f"{v.get(k, float('nan')):>{width}.3f}" for v in summaries.values())
        )


if __name__ == "__main__":
    main()
