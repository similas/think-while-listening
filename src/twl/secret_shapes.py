"""What a secret looks like: one list for the commit hook and the data deposit.

Responsibility: recognise token-shaped strings and credential-shaped file
names. ``src/scripts/secret_scan.py`` (the git hooks and the tree scan in
``make check``) uses both. ``src/scripts/release_bundle.py`` (the deposit)
uses the same TOKEN SHAPES, plus generic anchors such as ``authorization:``
that ordinary source code carries; it keeps its own, broader file-name rule
and skips binary files, which the hooks scan. Keeping the shapes here stops
the token lists drifting apart: a token shape the hooks refuse is one the
deposit refuses.

Invariants, tested in src/tests/test_secret_shapes.py:
- every shape is matched against LOWERCASED bytes, so each is written in lower
  case and a token's case cannot hide it;
- every match contains one of ``TOKEN_ANCHORS``, a literal prefilter that lets
  a caller skip the regex on text that cannot match;
- no pattern matches its own source text, so this module and its tests pass
  the tree scan; fake tokens in tests are assembled at runtime;
- a hit is reported by line and label, never by content.
"""

from __future__ import annotations

import re

# Label -> shape. Lower case throughout (see the module docstring).
TOKEN_SHAPES: dict[str, bytes] = {
    "overleaf_token": rb"\bolp_[a-z0-9]{20,}",
    "github_token": rb"\bgh[pousr]_[a-z0-9]{30,}",
    "github_pat": rb"\bgithub_pat_[a-z0-9_]{30,}",
    "huggingface_token": rb"\bhf_[a-z0-9]{30,}",
    "secret_key": rb"\bsk-[a-z0-9_-]{20,}",
    "live_key": rb"\b[sr]k_live_[a-z0-9]{20,}",
    "slack_token": rb"\bxox[abprs]-[a-z0-9-]{10,}",
    "aws_key": rb"\b(?:akia|asia)[0-9a-z]{16}\b",
    "gitlab_token": rb"\bglpat-[a-z0-9_-]{20,}",
    "google_api_key": rb"\baiza[0-9a-z_-]{35}",
    # With or without the PEM dashes: the deposit matched both before the
    # list was shared, and a header stripped of its dashes is still a key.
    "private_key": rb"begin [a-z ]*private key",
    # The scheme is bounded so a long run of letters cannot make this
    # quadratic: every start position costs at most 32 steps.
    "url_credentials": rb"[a-z][a-z0-9+.-]{0,31}://[^/\s:@]+:[^/\s@]{6,}@",
}
TOKEN_RE = re.compile(b"|".join(b"(?P<%s>%s)" % (k.encode(), v) for k, v in TOKEN_SHAPES.items()))
TOKEN_ANCHORS: tuple[bytes, ...] = (
    b"olp_",
    b"ghp_",
    b"gho_",
    b"ghu_",
    b"ghs_",
    b"ghr_",
    b"github_pat_",
    b"hf_",
    b"sk-",
    b"k_live_",
    b"xox",
    b"akia",
    b"asia",
    b"glpat-",
    b"aiza",
    b"private key",
    b"://",
)

# A file name that is a credential store whatever it holds. Matched against a
# repository-relative path with forward slashes.
SECRET_PATH = re.compile(
    r"(^|/)(\.env(\.[^/]*)?|\.netrc|\.git-credentials|[^/]*credentials(\.[^/]*)?"
    r"|id_(rsa|dsa|ecdsa|ed25519)(_[^/.]+)?)$"
    r"|(^|/)\.overleaf/"
    r"|\.(token|pem|p12|pfx)$",
    re.IGNORECASE,
)


def token_hits(text: bytes, first_line: int = 1) -> list[tuple[int, str]]:
    """Token-shaped strings in ``text``, as (line, label) pairs.

    Args:
        text: raw bytes; binary content is scanned too, since a token does not
            stop being one because the file around it holds a NUL.
        first_line: the line number of ``text``'s first line.

    Returns:
        One pair per match, in order. Never the matched content.
    """
    low = text.lower()
    if not any(a in low for a in TOKEN_ANCHORS):
        return []
    return [
        (first_line + low.count(b"\n", 0, m.start()), m.lastgroup or "")
        for m in TOKEN_RE.finditer(low)
    ]


def is_secret_path(path: str) -> bool:
    """True when ``path`` is named like a credential file."""
    return SECRET_PATH.search(path) is not None
