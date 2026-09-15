# think-while-listening

Resource-budgeted think-while-listening for on-device cascaded voice agents.

Cascaded voice agents (STT → LLM → TTS) waste the time during which the user is
still speaking; recent systems remove that dead time by reasoning speculatively on
partial transcripts — but all of them run on servers, where the background thinker
is free. On a unified-memory edge device the thinker and the listener share the
same silicon: speculative decode inflates the latency and word error rate of the
speech recognizer that is still transcribing the user, delays endpoint detection,
and can lengthen the very response it was meant to accelerate. This project
(i) characterizes that contention paradox on an NVIDIA Jetson Orin Nano Super,
(ii) introduces a controller that chooses, from live device state, *how much* to
think — a reasoning token budget — rather than merely *when*, and (iii) reports
the first latency–accuracy–energy Pareto for think-while-listening on a device
the user owns. When to think is largely solved; how much to think, on a device
that pays for it, is not.

## Reproducibility

Every number in `results/`, `paper/`, `thesis/`, `presentation/`, and this README
is produced by a script in `src/` from a log in `results/raw/`, and is rebuilt by
`make results` / `make figures`. Every results artifact records the git commit
hash, config hash, timestamp, device power mode, `jetson_clocks` status, and
software versions that produced it. Runs with any swap activity, OOM, or thermal
throttle event are recorded and flagged invalid, never silently dropped.

| target | builds |
|---|---|
| `make check` | ruff + mypy --strict + pytest (gate for every commit touching `src/`) |
| `make results` | all tables from `results/raw/` logs |
| `make figures` | all figures (vector PDF) from results |
| `make paper` | Interspeech paper (latexmk) |
| `make thesis` | Concordia M.Sc. thesis (PDF/A) |
| `make slides` | Beamer presentation |

See `CLAUDE.md` (project constitution), `RESEARCH_BRIEF_v2.md` (positioning,
metrics, hypotheses, protocol; wins where v1 disagrees), and `RESEARCH_BRIEF.md`
(v1; thesis format and presentation).
