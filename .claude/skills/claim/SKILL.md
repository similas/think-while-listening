---
name: claim
description: "Use whenever a number is to appear, change or disappear in paper/, thesis/, presentation/ or README.md. Covers registering the scorer line it comes from, generating its LaTeX macro, and regenerating the tables, RESULTS.md and the abstract check. Not for scoring a pass (score) or for the notebook entry (notes)."
---

# Put a number in the paper

No measured value is typed into LaTeX, the slides or the README. Every number
is one line of a scorer's output, registered in the claim registry, quoted
verbatim in `results/RESULTS.md` and turned into a macro in
`paper/generated/numbers.tex` and `thesis/generated/numbers.tex`.

## Order

1. The NOTES entry that records the number is COMMITTED (the `notes` skill).
   `commit_of()` resolves its heading with `git log -S` and fails on an
   uncommitted one. Commit first, then regenerate.
2. Register the SOURCE (the scorer invocation) in `src/scripts/paper_sources.py`,
   if it is new: an entry `"<key>": Source("<script>.py", (<args>, *RUNS)),`
   inside the `SOURCES` dict literal.
   Run ids are the ones NOTES gives, through `_runs("YYYYMMDD-HHMMSS-xxxxxx", ...)`.
   Pinned flags (`--until`, `--reps`, `--fill-p`) are part of the source.
3. Register the CLAIM:

   ```python
   Claim(
       "<key>",                      # snake_case, unique
       "<one-line statement, as the paper says it>",
       "<source key>",
       r"^<anchor of the ONE output line>",  # must match exactly one line
       "<exact NOTES heading, without '## '>",
   ),
   ```

   A pattern that matches no line or more than one stops the build. Anchor on
   the line's fixed prefix (`r"^ALLOCATOR decisions"`), and on a separator
   when a word recurs: the `break_even` macro once matched "below_break_even";
   it is now anchored as `r"; break_even ([0-9.]+)"`.
4. Register the MACRO in `src/scripts/paper_results.py`:
   - a value as printed: add `("<Macro>", "<claim key>", r"<group>")` to the
     `NUMBERS` tuple;
   - a value re-formatted: add `("<Macro>", "<claim key>", r"<groups>", "<format>")`
     to `FORMATTED`, e.g. format `"{0:+.1f} ms [{1:+.1f}, {2:+.1f}]"`.
   Macro names are CamelCase letters only, since LaTeX macros take no digits:
   `PthreeRed`, `PnineKill`, `SttShareMulti`. A number and its n are separate
   macros (`PthreeRed`, `PthreeN`).
5. `make results`. It runs every source, quotes each claim line in RESULTS.md,
   writes both `generated/numbers.tex`, the tables and the appendices, and
   checks the paper abstract against `ABSTRACT_MAX_CHARS = 1000` (Interspeech
   2027, characters with spaces, macros expanded). It stops on any macro it
   cannot expand.
6. Use the macro in the text: `\PthreeRed{} ($n=\PthreeN$)`. Every number in
   the text carries its n and CI where it has them.
7. Commit the registry, the generated files and the text together. A change
   to a figure, table or macro means a compile of the paper and the thesis is
   OWED, reporting the page count under `\extendedfalse` and any overfull-box
   or missing-package warning (Ali, step 7 of the learned-committer phase).
   LaTeX is never compiled on the box, and how the compile happens is pending
   Ali (the `overleaf` skill). Say in the report that a compile is owed; never
   claim one happened.

## Checks before the commit

- `grep` the changed .tex for a typed digit next to "ms", "%", "x", "n=": each
  must be a macro.
- RESULTS.md shows the claim's line verbatim, with its scorer and run ids.
- In the thesis, a number with no claim yet is written as the red box
  `\needsclaim{<what>}{<NOTES heading>}`, never as a typed value. The thesis
  is not final while one remains.
- `make check` if anything under `src/` changed.
