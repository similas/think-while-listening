"""LocalAgreement commitment: what the listener can keep before the endpoint.

Responsibility: own one turn's COMMITTED transcript. Words two consecutive
hypotheses agree on are committed and never revisited; the audio behind them is
never decoded again. At the endpoint the final decodes only what is left, which
is the whole point — the final is 57-83 % of TTFA on this device and its cost is
linear in the audio it is handed.

Algorithm: LocalAgreement-n (Machacek, Dabre & Bojar 2023, `whisper_streaming`).
Keep the last ``agreement_n - 1`` hypotheses' unagreed remainders; commit the
longest common prefix across them and the newest hypothesis.

Invariants, all tested in src/tests/test_commit.py:
- committed words never duplicate across a commit boundary;
- ``committed_end_s`` is monotone;
- a word ending inside the tail guard is never committed, because the buffer
  edge cuts words and a cut word is a wrong word committed for ever;
- the comparison is on NORMALIZED tokens, but what is committed is the
  hypothesis's own text and timestamps — normalization decides agreement, it
  does not become the transcript.

This module is pure and synchronous. It holds no audio, starts no decode and
knows nothing about engines; everything that can be decided without the
recognizer is decided here, where it can be tested exhaustively.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PUNCT = re.compile(r"[^\w']+")


@dataclass(frozen=True)
class Word:
    """One decoded word, timed in the TURN's audio, not the buffer's.

    The recognizer sees a trimmed buffer starting at ``committed_end_s``, so
    its timestamps are buffer-relative and the caller re-bases them before
    offering them here. Keeping turn-absolute times is what lets a hypothesis
    decoded on a trimmed buffer be compared with one decoded on an earlier,
    longer buffer at all.
    """

    text: str
    start_s: float
    end_s: float

    @property
    def key(self) -> str:
        """The token used for AGREEMENT — lowercase, punctuation stripped.

        "four." and "four" are the same word arriving with different
        punctuation from two decodes of overlapping audio; treating them as
        different would stall the commit for ever.
        """
        return _PUNCT.sub("", self.text.lower())


@dataclass
class LocalAgreementCommitter:
    """Commits the prefix that successive hypotheses agree on."""

    agreement_n: int = 2
    tail_guard_s: float = 0.3
    committed: list[Word] = field(default_factory=list)
    committed_end_s: float = 0.0
    # The unagreed tails of the last agreement_n - 1 hypotheses, newest last.
    _remainders: list[list[Word]] = field(default_factory=list, repr=False)

    def offer(self, hyp: list[Word], buffer_end_s: float) -> list[Word]:
        """Feed one hypothesis decoded on audio[committed_end_s : buffer_end_s].

        Args:
            hyp: the hypothesis's words, timed in the turn's audio.
            buffer_end_s: how far into the turn the decoded buffer reached.

        Returns:
            The words committed by THIS call, in order; possibly empty.
        """
        fresh = [w for w in hyp if w.start_s >= self.committed_end_s - 1e-9]
        prefix = self._agreed_prefix(fresh)
        guard = buffer_end_s - self.tail_guard_s
        # A word the buffer edge may have cut is not committed, however much
        # the hypotheses agree: they agree because they saw the same truncation.
        keep: list[Word] = []
        for w in prefix:
            if w.end_s > guard:
                break
            keep.append(w)
        if keep:
            self.committed.extend(keep)
            self.committed_end_s = max(self.committed_end_s, keep[-1].end_s)
        # The remainder is what THIS hypothesis said beyond what is now
        # committed — the material the next hypothesis has to agree with.
        self._remainders.append([w for w in fresh if w.start_s >= self.committed_end_s - 1e-9])
        excess = len(self._remainders) - (self.agreement_n - 1)
        if excess > 0:
            del self._remainders[:excess]
        return keep

    def _agreed_prefix(self, hyp: list[Word]) -> list[Word]:
        """Longest common token prefix of the stored remainders and ``hyp``."""
        if len(self._remainders) < self.agreement_n - 1:
            return []
        others = [[w.key for w in r] for r in self._remainders]
        out: list[Word] = []
        for i, w in enumerate(hyp):
            if any(i >= len(o) or o[i] != w.key for o in others):
                break
            out.append(w)
        return out

    def committed_text(self) -> str:
        return " ".join(w.text for w in self.committed).strip()

    def prompt(self, max_chars: int = 200) -> str:
        """The tail of the committed text, for the recognizer's initial_prompt.

        A trimmed buffer has no left context, so the decoder is guessing at a
        sentence it cannot see the start of. The tail is given back as prompt
        text; it is CONTEXT, never output, and is not part of the transcript.
        """
        text = self.committed_text()
        return text[-max_chars:] if len(text) > max_chars else text

    def reset(self) -> None:
        self.committed = []
        self.committed_end_s = 0.0
        self._remainders = []
