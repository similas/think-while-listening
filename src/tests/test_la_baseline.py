"""LocalAgreement-2's operating point on the labels (twl.la_baseline)."""

from __future__ import annotations

from twl.la_baseline import is_number, la2_counts, runaway_hyps, simulated_commits


def row(
    lab: int,
    commit: int,
    hyp: int = 1,
    start: float = 0.0,
    key: str = "w",
    turn: int = 0,
    index: int = 0,
) -> dict[str, object]:
    return {
        "label_num": lab,
        "la2_commit": commit,
        "turn": turn,
        "hyp_index": hyp,
        "word_index": index,
        "start_s": start,
        "end_s": start + 0.4,
        "key": key,
    }


def test_la2_counts_split_declines_into_later_committed_and_never() -> None:
    rows = [
        row(1, 0, hyp=1, start=0.0, key="one"),  # declined, committed at hyp 2
        row(1, 1, hyp=2, start=0.05, key="one"),
        row(0, 0, hyp=1, start=0.5, key="two"),  # declined, never committed
        row(1, 0, hyp=2, start=0.5, key="two"),
        row(0, 1, hyp=2, start=1.0, key="three"),  # a false commit
        row(1, 1, hyp=1, start=2.0, key="x", turn=1),  # another turn
        row(-1, 1),  # unscored
    ]
    c = la2_counts(rows, final_words=10)
    assert (c.committed, c.committed_survive) == (3, 2)
    assert (c.declined, c.declined_survive) == (3, 2)
    assert (c.declined_later, c.declined_later_survive) == (1, 1)
    assert c.final_words == 10


def test_a_later_commit_must_be_later_close_in_time_and_in_the_same_turn() -> None:
    rows = [
        row(1, 0, hyp=2, start=0.0, key="a"),
        row(1, 1, hyp=1, start=0.0, key="a"),  # earlier hypothesis: not "later"
        row(1, 0, hyp=1, start=5.0, key="b"),
        row(1, 1, hyp=2, start=6.0, key="b"),  # too far in time
        row(1, 0, hyp=1, start=0.0, key="c", turn=0),
        row(1, 1, hyp=2, start=0.0, key="c", turn=1),  # another turn
    ]
    assert la2_counts(rows, 6).declined_later == 0


def test_runaway_hypotheses_are_those_stuck_on_one_token() -> None:
    stuck = [row(0, 0, hyp=3, key="phi") for _ in range(15)] + [
        row(0, 0, hyp=3, key=f"k{i}") for i in range(10)
    ]
    normal = [row(1, 0, hyp=4, key=f"w{i}") for i in range(30)]
    short = [row(0, 0, hyp=5, key="phi") for _ in range(10)]
    assert runaway_hyps(stuck + normal + short) == {(0, 3)}


def test_number_words() -> None:
    assert is_number("12") and is_number("seven") and is_number("$1,000")
    assert not is_number("cookies") and not is_number("percent")


def words(*spec: tuple[str, float]) -> list[dict[str, object]]:
    return [{"text": t, "start_s": s, "end_s": s + 0.4} for t, s in spec]


def test_simulated_commits_key_by_position_even_at_identical_times() -> None:
    same = words(("north", 1.0), ("north", 1.0), ("north", 1.0))
    hyps: list[dict[str, object]] = [
        {"frozen": False, "issued_s": 3.0, "committed_end_s": 0.0, "words": same},
        {"frozen": False, "issued_s": 4.0, "committed_end_s": 0.0, "words": same},
    ]
    assert simulated_commits(hyps, 2, 0.3) == {(2, 0), (2, 1), (2, 2)}


def test_simulated_la2_and_la3_and_frozen_hypotheses() -> None:
    a = words(("one", 0.0), ("two", 0.5))
    hyps: list[dict[str, object]] = [
        {"frozen": False, "issued_s": 2.0, "committed_end_s": 0.0, "words": a},
        {"frozen": False, "issued_s": 3.0, "committed_end_s": 0.0, "words": a},
        {"frozen": True, "issued_s": 3.5, "committed_end_s": 0.0, "words": []},
        {"frozen": False, "issued_s": 4.0, "committed_end_s": 0.0, "words": a},
    ]
    assert simulated_commits(hyps[:2], 2, 0.3) == {(2, 0), (2, 1)}
    assert simulated_commits(hyps[:2], 3, 0.3) == set()
    assert simulated_commits(hyps, 3, 0.3) == {(3, 0), (3, 1)}
