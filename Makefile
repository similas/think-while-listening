# All quality gates and artifacts build from here. `make check` must pass before
# every commit that touches src/ (CLAUDE.md §3).

VENV ?= $(HOME)/.venvs/twl
BIN  := $(VENV)/bin

.PHONY: check results figures paper thesis slides

check:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests
	$(BIN)/mypy
	$(BIN)/pytest -q

# The four artifact targets are introduced by later phases. Until their inputs
# exist, building them is a misconfiguration and fails loudly (CLAUDE.md §3).
results:
	@echo "error: no result scripts yet — introduced in Phase 1 (src/twl + results/raw)" >&2; exit 2

figures:
	@echo "error: no figure scripts yet — introduced in Phase 2" >&2; exit 2

paper:
	@echo "error: paper/ sources not written yet — introduced in Phase 6" >&2; exit 2

thesis:
	@echo "error: thesis/ sources not written yet — introduced in Phase 7" >&2; exit 2

slides:
	@echo "error: presentation/ sources not written yet — introduced in Phase 7" >&2; exit 2
