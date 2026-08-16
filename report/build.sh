#!/usr/bin/env bash
# Build the LaTeX report with XeLaTeX.
# Usage: bash report/build.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main || true
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex

echo "[ok] report/main.pdf built"