#!/usr/bin/env bash
# =============================================================================
# run_all.sh — End-to-end NLP Assignment Pipeline
#
# Usage:
#   bash scripts/run_all.sh
#
# Module outputs (as required by specification):
#   output/clauses.txt
#   output/chunks.txt
#   output/dependency.json
#   output/ner_results.json
#   output/srl_results.json
#   output/intent_classification.txt
#
# Organised copies (per-assignment, for easy review):
#   results/assignment1/clauses.txt
#   results/assignment1/chunks.txt
#   results/assignment1/dependency.json
#   results/assignment2/ner_results.json
#   results/assignment2/srl_results.json
#   results/assignment2/intent_classification.txt
#
# Assignment 3 (RAG chatbot) is commented out — run manually when ready.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

info "Project root: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 1. Python / virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"

if [ ! -f "$PYTHON" ]; then
    info "Creating virtual environment at .venv ..."
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created."
else
    info "Using existing virtual environment at .venv"
fi

info "Activating virtual environment ..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
info "Installing Python dependencies from requirements.txt ..."
pip install --quiet --upgrade pip
pip install --quiet -r "$PROJECT_ROOT/requirements.txt"
ok "Dependencies installed."

# ---------------------------------------------------------------------------
# 3. Download spaCy model
# ---------------------------------------------------------------------------
if ! "$PYTHON" -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    info "Downloading spaCy model en_core_web_sm ..."
    "$PYTHON" -m spacy download en_core_web_sm --quiet
    ok "spaCy model downloaded."
else
    info "spaCy model en_core_web_sm already installed."
fi

# ---------------------------------------------------------------------------
# 4. Prepare directories
# ---------------------------------------------------------------------------
mkdir -p "$PROJECT_ROOT/output"
mkdir -p "$PROJECT_ROOT/results/assignment1"
mkdir -p "$PROJECT_ROOT/results/assignment2"
# mkdir -p "$PROJECT_ROOT/results/assignment3"   # Assignment 3 — uncomment when ready

# ---------------------------------------------------------------------------
# 5. Assignment 1 — Preprocessing & Syntax Analysis
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  ASSIGNMENT 1 — Preprocessing & Syntax Analysis${NC}"
echo -e "${BOLD}============================================================${NC}"

info "Running Assignment 1 ..."
"$PYTHON" src/assignment1/run_assignment1.py
ok "Assignment 1 complete."

# Copy spec-required outputs → results/assignment1/
info "Copying Assignment 1 results to results/assignment1/ ..."
for f in clauses.txt chunks.txt dependency.json; do
    [ -f "$PROJECT_ROOT/output/$f" ] && cp "$PROJECT_ROOT/output/$f" "$PROJECT_ROOT/results/assignment1/$f"
done

# ---------------------------------------------------------------------------
# 6. Assignment 2 — Information Extraction & Semantic Analysis
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  ASSIGNMENT 2 — Information Extraction & Semantic Analysis${NC}"
echo -e "${BOLD}============================================================${NC}"

info "Running Assignment 2 ..."
"$PYTHON" src/assignment2/run_assignment2.py
ok "Assignment 2 complete."

# Copy spec-required outputs → results/assignment2/
info "Copying Assignment 2 results to results/assignment2/ ..."
for f in ner_results.json srl_results.json intent_classification.txt; do
    [ -f "$PROJECT_ROOT/output/$f" ] && cp "$PROJECT_ROOT/output/$f" "$PROJECT_ROOT/results/assignment2/$f"
done

# ---------------------------------------------------------------------------
# 7. Assignment 3 — RAG Chatbot (BONUS) — COMMENTED OUT
# ---------------------------------------------------------------------------
# Uncomment when ready. Requires GOOGLE_API_KEY set in .env
#
# echo ""
# echo -e "${BOLD}============================================================${NC}"
# echo -e "${BOLD}  ASSIGNMENT 3 — RAG Chatbot (Bonus)${NC}"
# echo -e "${BOLD}============================================================${NC}"
# if [ ! -f "$PROJECT_ROOT/.env" ]; then
#     error ".env file missing — copy .env.example → .env and set GOOGLE_API_KEY."
#     exit 1
# fi
# info "Indexing clauses into vector store ..."
# "$PYTHON" -c "
# import sys; sys.path.insert(0, 'src')
# from assignment3.vector_store import LegalContractVectorStore
# vs = LegalContractVectorStore()
# vs.index_clauses('output/intent_classification.txt')
# print('Vector store indexed.')
# "
# mkdir -p "$PROJECT_ROOT/results/assignment3"
# ok "Assignment 3 vector store ready."
# info "Starting Streamlit app (Ctrl+C to stop) ..."
# "$PYTHON" -m streamlit run src/assignment3/app.py

# ---------------------------------------------------------------------------
# 8. Final summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  PIPELINE COMPLETE${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "  ${BOLD}Spec-required outputs (output/)${NC}"
for f in clauses.txt chunks.txt dependency.json ner_results.json srl_results.json intent_classification.txt; do
    fp="$PROJECT_ROOT/output/$f"
    if [ -f "$fp" ]; then
        size_kb=$(du -k "$fp" | cut -f1)
        printf "  ${GREEN}✓${NC}  %-45s %4s KB\n" "output/$f" "$size_kb"
    else
        printf "  ${RED}✗${NC}  %-45s (missing)\n" "output/$f"
    fi
done

echo ""
echo -e "  ${BOLD}Organised copies (results/)${NC}"
for entry in \
    "assignment1/clauses.txt" \
    "assignment1/chunks.txt" \
    "assignment1/dependency.json" \
    "assignment2/ner_results.json" \
    "assignment2/srl_results.json" \
    "assignment2/intent_classification.txt"
do
    fp="$PROJECT_ROOT/results/$entry"
    if [ -f "$fp" ]; then
        size_kb=$(du -k "$fp" | cut -f1)
        printf "  ${GREEN}✓${NC}  %-45s %4s KB\n" "results/$entry" "$size_kb"
    else
        printf "  ${RED}✗${NC}  %-45s (missing)\n" "results/$entry"
    fi
done

echo ""
ok "All done!"

