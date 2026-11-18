# Switching paper/main.tex to the Interspeech 2027 kit

The draft is written against `article` so it builds anywhere. When the 2027
author kit arrives, these are the places the kit's class may differ from the
current template. Each item says what the draft does now and what to check
in the kit. Nothing here is a claim about the 2027 kit; it is a checklist.

## Build switches (keep)

| build | `\anonymous` | `\extended` |
|---|---|---|
| Interspeech submission | true | false |
| Interspeech camera-ready | false | false |
| arXiv extended | false | true |

## Preamble

1. `\documentclass[a4paper,10pt,twocolumn]{article}` → the kit's class. It
   sets the paper size, fonts, margins, column layout and section styling.
   Delete the options and check that nothing below relies on them.
2. `inputenc`, `amsmath`, `amssymb`, `graphicx`, `booktabs`, `longtable`,
   `xcolor`, `hyperref`: check which the kit loads itself, which it forbids
   (hyperref is the usual one), and with what options. `longtable` is used only
   in the extended build's one-column appendix.
3. `\hypersetup{pdfauthor=...}`: if the kit forbids hyperref, drop both
   branches. Check that the anonymous PDF's metadata carries no author (also
   check the kit's own metadata macros).
4. `\graphicspath{{figures/}}` and `paper/figures → ../results/figures`: check
   that the kit's submission system accepts a symlinked directory. Otherwise
   copy the five `paper_fig*.pdf` into the upload.

## Title block

5. `\title`, `\author`, `\date`, `\maketitle`: the kit has its own author and
   affiliation macros. Map the two `\ifanonymous` branches onto them.
   Anonymous: no name, affiliation or email. Camera-ready: Ali Salimi Sadr,
   Concordia University, Montreal, Canada, plus email if the kit asks.
6. Keywords: the draft has none. Check whether the kit requires an index-terms
   or keywords block after the abstract.
7. Abstract: `\begin{abstract}` in the draft. Check the kit's environment and
   any word limit, and re-count with the macros expanded.

## Body

8. Figures 2 and 5 are `figure*` (full width), 1, 3 and 4 are `figure`.
   Tables 1 and 2 are generated as `table*`; the six-row ledger summary is
   generated as `table*`. Check the kit's float placement rules and caption
   style (caption above or below tables).
9. Section numbering and `\subsection` depth: the Results section has ten
   subsections. Check whether the kit numbers subsections and how deep.
10. The two `\pendingpassnine` boxes, in Results ("Live speech") and in
    Limitations ("Audio"), must be replaced before submission.

## The +2 pages

11. References: `\bibliographystyle{IEEEtran}` + `refs.bib`. Check the kit's
    style file name, and whether it wants `\bibliography` or its own command.
12. After the references, the draft carries two unnumbered sections: the
    Reproducibility statement and the Generative-AI disclosure (Ali, 2026-11-18:
    the disclosure moves to the +2 pages with the references). Check that the
    2027 policy allows these on the reference pages, and whether it prescribes
    a heading or a fixed position for the generative-AI statement.
13. Acknowledgments: an unnumbered section before the references, stripped
    when anonymous. Check where the kit wants it and whether it counts toward
    the page limit.

## Anonymity (`\anonymoustrue`)

14. Stripped: author name, affiliation, PDF author metadata, acknowledgments,
    the repository URL and the Zenodo DOI. The reproducibility statement then
    says the release follows publication.
15. Still present by design, not identifying: the device model, the
    generative-AI disclosure (it names a tool, not the author), the
    Spoken-MQA citation.
16. Check before upload: (a) the PDF metadata (`pdfinfo`); (b) that no figure
    PDF embeds a path or user name (`strings paper_fig*.pdf | grep -i ali`);
    (c) that `refs.bib`'s header comment does not reach the PDF; (d) that the
    supplementary material, if any, is anonymized the same way.

## Extended build (`\extendedtrue`)

`\anonymoustrue` together with `\extendedtrue` stops the build with an error:
the extended appendix quotes the notebook, which names the author.

17. It adds the planner's formulation and calibration to "Is the gate enough?",
    and a one-column appendix holding the full ledger (longtable), the
    retractions log (longtable) and the capture-path hazards. arXiv takes
    `article` as is; none of the kit items above apply to it.
