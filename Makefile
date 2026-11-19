# All quality gates and artifacts build from here. `make check` must pass before
# every commit that touches src/ (CLAUDE.md §3).
#
# LaTeX is never compiled on the Jetson: texlive would cost memory and disk the
# experiments need. paper/, thesis/ and presentation/ hold sources only; the
# paper/thesis/slides targets run on LATEX_HOST (Ali's Mac, or Overleaf).

VENV ?= $(HOME)/.venvs/twl
BIN  := $(VENV)/bin
LATEX_HOST ?= mac

.PHONY: check hooks results figures paper thesis slides release

check:
	PYTHONPATH=src $(BIN)/python src/scripts/secret_scan.py --tree
	$(BIN)/ruff check src
	$(BIN)/ruff format --check src
	$(BIN)/mypy
	$(BIN)/pytest -q

# Install the secret-scan hooks (pre-commit, pre-merge-commit, pre-push) into
# this clone's .git/hooks and, when it exists, the Overleaf clone's: that clone
# pushes with the Overleaf token. Once per clone (the Mac too).
OVERLEAF_CLONE ?= $(HOME)/.cache/twl-overleaf/paper
hooks:
	src/scripts/hooks/install.sh "$(CURDIR)" "$(OVERLEAF_CLONE)"

# The artifact targets are introduced by later phases. Until their inputs exist,
# building them is a misconfiguration and fails loudly (CLAUDE.md §3).
results:
	PYTHONPATH=src $(BIN)/python src/scripts/make_tables.py
	PYTHONPATH=src $(BIN)/python src/scripts/trigger_auc.py
	PYTHONPATH=src $(BIN)/python src/scripts/audit_gpu_offload.py
	PYTHONPATH=src $(BIN)/python src/scripts/score_prefix_probe.py
	PYTHONPATH=src $(BIN)/python src/scripts/final_decode_model.py
	PYTHONPATH=src $(BIN)/python src/scripts/stage_decomposition.py
	PYTHONPATH=src $(BIN)/python src/scripts/energy_per_turn.py
	PYTHONPATH=src $(BIN)/python src/scripts/wer_by_arm.py
	PYTHONPATH=src $(BIN)/python src/scripts/paper_results.py

figures:
	PYTHONPATH=src $(BIN)/python src/scripts/make_figures.py
	PYTHONPATH=src $(BIN)/python src/scripts/paper_figures.py

# LaTeX targets run on LATEX_HOST only. Each builds from its own directory, so
# the generated/ inputs and the figures symlink resolve.
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

paper thesis slides:
	@command -v latexmk >/dev/null 2>&1 || { \
	  echo "error: latexmk not found — '$@' compiles on $(LATEX_HOST), never on the Jetson" >&2; exit 2; }
	cd $(if $(filter slides,$@),presentation,$@) && $(LATEXMK) main.tex

# The data deposit (logs without audio, checksums, claim manifest). Built and
# published after notification; reviewers use the anonymized mirror until then.
release:
	PYTHONPATH=src $(BIN)/python src/scripts/release_bundle.py
