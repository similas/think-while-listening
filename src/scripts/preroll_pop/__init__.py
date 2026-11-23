"""Per-scorer turn populations for the pre-roll audit (preroll_exposure.py).

One module per registered scorer that reads live turn records. Each exposes
``population() -> dict[str, set[tuple[str, int]]]`` keyed by claim (keys with
"::" or "__" or a dotted suffix are gates and breakdowns), computed by calling
the scorer's own functions or by a verbatim copy of an inline filter, cited by
line. Run as __main__, each prints its unit counts beside the counts the
scorer prints (2026-11-22).
"""

import re


def numbers(text: str) -> list[str]:
    """The signed numbers in a recomputed fragment, each to be found in the
    scorer's line as a whole token."""
    return re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\d.])", text)


def check(claim: str, *fragments: str) -> str:
    """Assert each fragment appears in the registry's line for ``claim`` (the
    scorer run as paper_sources runs it); return that line."""
    from scripts.paper_sources import CLAIMS, line

    c = next(c for c in CLAIMS if c.key == claim)
    text = line(c.source, c.pattern).string
    for f in fragments:
        # delimited: "n=21" must not pass against "n=216"
        hit = re.search(re.escape(f) + (r"(?![\d.])" if f[-1:].isdigit() else ""), text)
        assert hit, f"{claim}: {f!r} not in the scorer's line {text!r}"
    print(f"verified {claim}: {', '.join(fragments)}")
    return text
