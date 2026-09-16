# All quality gates and artifacts build from here. `make check` must pass before
# every commit that touches src/ (CLAUDE.md §3).
#
# LaTeX is never compiled on the Jetson: texlive would cost memory and disk the
# experiments need. paper/, thesis/ and presentation/ hold sources only; the
# paper/thesis/slides targets run on LATEX_HOST (Ali's Mac, or Overleaf).

VENV ?= $(HOME)/.venvs/twl
BIN  := $(VENV)/bin
LATEX_HOST ?= mac

.PHONY: check results figures paper thesis slides

check:
	$(BIN)/ruff check src
	$(BIN)/ruff format --check src
	$(BIN)/mypy
	$(BIN)/pytest -q

# The artifact targets are introduced by later phases. Until their inputs exist,
# building them is a misconfiguration and fails loudly (CLAUDE.md §3).
results:
	PYTHONPATH=src $(BIN)/python src/scripts/make_tables.py

figures:
	PYTHONPATH=src $(BIN)/python src/scripts/make_figures.py

paper thesis slides:
	@command -v latexmk >/dev/null 2>&1 || { \
	  echo "error: latexmk not found — '$@' compiles on $(LATEX_HOST), never on the Jetson" >&2; exit 2; }
	@echo "error: $@/ sources not written yet — introduced in Phase 6/7" >&2; exit 2
