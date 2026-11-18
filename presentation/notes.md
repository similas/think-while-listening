# Presentation: build and provenance

`main.tex` is the talk: Beamer, metropolis, 16:9, 21 slides plus 2 backup
slides. Speaker notes are `\note{}` on every slide.

## Build (on the LaTeX host, never on the Jetson)

    cd presentation
    latexmk -xelatex main.tex      # metropolis's fonts; or: latexmk -pdf main.tex

It reads `../paper/generated/numbers.tex`, `../paper/generated/table*.tex` and
`../paper/figures/` (the symlink to `results/figures`). Run `make results`
first so the macros and tables are current. Packages: beamer, metropolis,
booktabs, pgfpages, amsmath.

Speaker notes on a second screen: uncomment
`\setbeameroption{show notes on second screen=right}` in the preamble. The PDF
then carries the slide on the left half of each page and its note on the right.

## Provenance

Every measured or simulated value on a slide or in a note is a macro from
`paper/generated/numbers.tex`. Typed: registered bars and predictions, set
sizes, config constants, labels (pass and prediction names).

`\NoMacro{n}` prints `[n: no macro]`. It marks an n the note needs and
`numbers.tex` does not carry. To close one, add the macro in
`src/scripts/paper_results.py`, regenerate, and replace the marker. Never type
the value.

Open markers (`grep -n NoMacro main.tex`): the n of `\PnineKill`,
`\PnineRatio`, `\DevTTFA`, `\PfourMean`/`\PfourCI`, `\PsixRatio`,
`\EnergyListener`/`\EnergyListenerr`, `\SevenJms`, `\SpecEnergyr`,
`\PlannerDP`/`\PlannerNoDuty`/`\PlannerVal`, `\WorkerPSS`, `\ResidencyTTFA`,
`\ResidencyMem`.

Left out because no macro carries them (each is described in words, or read
off a figure, instead): the turn counts and per-set TTFA of Fig 1; the
listener's in-flight bar of Fig 3 and the median overlap behind it; the P2a
separate-arm coding; f* as first registered on mixed clocks; the planner's
mean gain, its count of items gaining at least 100 ms, and the value measured
on its items; the per-reason inputs of Table 2; the MemAvailable drop at worker
spawn; the base rate behind the trigger AUCs.

## Generated tables on a 16:9 slide

- `table1.tex`: does not fit. The slide "Where to speculate" is a compact
  rebuild from the same macros; the generated table is the first backup slide,
  scaled by `\twlgenerated`, for zooming, not projecting.
- `table2.tex`: nine columns, does not fit legibly. The slide "The allocator
  never spent the window on a draft" rebuilds its counts from the macros; the
  generated table is the second backup slide.
- `table3_summary.tex`: used as generated on "The record", scaled to the frame.
- `table3.tex` (the full ledger): does not fit; not used.

## First compile: check by eye

The deck has never been compiled. Look for overflow on the figure slides with
two or three bullets (Figs 1, 3, 4, 5), "Testbed", "Where to speculate" and
"The record", and check that `\twlgenerated` renders the three generated
tables.
