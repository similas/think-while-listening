---
name: overleaf
description: "Use only when Ali asks to push the paper to Overleaf or to bring Ali's Overleaf edits back into the repository. Covers the bundle, the pull-before-push rule, the commit and the token's handling. LaTeX is never compiled on the box."
disable-model-invocation: true
---

# Overleaf

The repository is the source of truth. Overleaf holds a BUILT BUNDLE of the
paper, so it can compile there and Ali can read and edit it. The direction is
one way, repository → Overleaf. Ali's edits on Overleaf come back into the
repository before the next push, so they are never overwritten.

| | |
|---|---|
| project | `https://www.overleaf.com/project/6afe31243c10ba01d9ef5ec7` (paper only) |
| git remote | `https://git.overleaf.com/6afe31243c10ba01d9ef5ec7`, user `git` |
| local clone | `~/.cache/twl-overleaf/paper`, user name and email from this repository's git config |
| hooks | `make hooks` writes secret-scan stubs into the clone's `.git/hooks` (pre-commit, pre-merge-commit, pre-push), calling this repository's `src/scripts/hooks`: its commits and pushes are scanned |
| token | `~/.config/overleaf/credentials`, mode 600, read by `credential.https://git.overleaf.com.helper=store --file ...` in `~/.gitconfig` |
| bundle | `dist/overleaf/paper/` (gitignored) |

THE TOKEN: never print it, paste it, put it in a URL, a command line, a commit,
a NOTES entry or a report. If it is ever seen in a session, Ali revokes it and
writes a new one into the credentials file by hand, outside any session.

## 1. Pull first, always, and bring Ali's edits back

```bash
O=~/.cache/twl-overleaf/paper
git -C $O pull --ff-only
# the last commit we pushed names the repository commit it was built from
LAST=$(git -C $O log --format=%H --grep='^Paper: .* from [0-9a-f]\{7,\}' -1)
LAST=${LAST:-947b699}                      # the first push predates the "from <sha>" message
git -C $O log --oneline $LAST..HEAD        # anything here is an Overleaf edit
git -C $O diff $LAST..HEAD -- main.tex refs.bib > /tmp/overleaf-edits.patch
if [ -s /tmp/overleaf-edits.patch ]; then
  git apply --check --directory=paper /tmp/overleaf-edits.patch && git apply --directory=paper /tmp/overleaf-edits.patch
else
  echo "no Overleaf edits"
fi
```

Overleaf's own commit message for Ali's edits is "Update on Overleaf.". The
diff runs from our last push to Overleaf's head, so it holds only Ali's edits;
it never reverts repository work done since. A hunk that does not apply is
resolved by hand. Commit the edits in this repository before anything else.
An edit that types a number is not taken as is: the number goes back through
the `claim` skill. Files under `generated/` and `figures/` are never edited on
Overleaf. They are regenerated from the repository. The first push,
947b699 on 2026-11-18 ("Paper: sources, generated tables and figures from the
repository"), predates the "from <sha>" convention, hence the fallback.

## 2. Build the bundle

Real files, not the repository's symlink (`paper/figures → ../results/figures`):

```bash
rm -rf dist/overleaf/paper && mkdir -p dist/overleaf/paper/{generated,figures}
cp paper/main.tex paper/refs.bib dist/overleaf/paper/
cp paper/generated/*.tex dist/overleaf/paper/generated/
cp results/figures/paper_fig[1-5]_*.pdf dist/overleaf/paper/figures/
```

The build switches stay as committed (`\anonymous`, `\extended`; see the
header of `main.tex`). The anonymous submission is never pushed with author
metadata (`paper/INTERSPEECH_KIT_NOTES.md`, item 16).

## 3. Commit and push to Overleaf

```bash
diff -rq --exclude=.git dist/overleaf/paper $O | grep "^Only in $O"   # files only Overleaf has: Ali's; ask before removing
rsync -a --exclude .git dist/overleaf/paper/ $O/
git -C $O add -A
git -C $O commit -m "Paper: <what changed>, from $(git rev-parse --short HEAD)"
git -C $O push
```

The message must end "from <repository sha>": step 1 finds the last push by
it. No attribution lines. Then report the repository sha pushed and what
changed.

## Compiling

LaTeX is never compiled on the box. Overleaf compiles in Ali's browser; its
log and page count are not readable from here. Ali's step 7 (2026-11-18) asks
for a compile of the paper and the thesis after every change to a figure,
table or macro, reporting the page count under `\extendedfalse` and any
overfull-box or missing-package warnings, "through the Overleaf-GitHub sync".
In place is this git-bridge clone, not the GitHub sync, and neither reports
back to the box. HOW is pending Ali. Until Ali decides, say after each such
change that a compile is owed, and never claim one happened. The thesis has
no Overleaf project yet.
