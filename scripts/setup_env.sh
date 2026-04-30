#!/usr/bin/env bash
# =============================================================================
# NLP Assignment — Environment Setup Script
# Creates a Python virtual environment and installs all required dependencies
# for the bilingual (EN + VI) legal contract NLP pipeline + RAG chatbot.
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}  ℹ  $*${NC}"; }
success() { echo -e "${GREEN}  ✓  $*${NC}"; }
warn()    { echo -e "${YELLOW}  ⚠  $*${NC}"; }
error()   { echo -e "${RED}  ✗  $*${NC}"; }
header()  { echo -e "\n${BOLD}${GREEN}$*${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"
ENV_FILE="$PROJECT_ROOT/.env"
REQ_FILE="$PROJECT_ROOT/requirements.txt"

header "=== NLP Assignment — Environment Setup ==="
info "Project root : $PROJECT_ROOT"
info "Virtualenv   : $VENV_DIR"

# ── Python version check ──────────────────────────────────────────────────────
if command -v python3.11 &>/dev/null; then
    PYTHON="python3.11"
elif command -v python3.10 &>/dev/null; then
    PYTHON="python3.10"
elif command -v python3.9 &>/dev/null; then
    PYTHON="python3.9"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    error "Python 3.9+ required but not found."
    exit 1
fi
info "Using: $PYTHON ($($PYTHON --version 2>&1))"

# ── Create virtual environment ────────────────────────────────────────────────
header "[1/7] Creating virtual environment"
if [ -d "$VENV_DIR" ]; then
    warn "Virtualenv already exists at $VENV_DIR — skipping creation."
else
    $PYTHON -m venv "$VENV_DIR"
    success "Virtualenv created → $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# Upgrade pip/setuptools silently
"$VENV_PIP" install --quiet --upgrade pip setuptools wheel

# ── Install Python dependencies ───────────────────────────────────────────────
header "[2/7] Installing Python packages"
info "This may take 5–15 minutes on first run..."

# Install PyTorch first via the official wheel index (CPU build).
# requirements.txt lists torch>=2.2 which pip will treat as satisfied once
# torch is already present, regardless of the index URL used to fetch it.
# To use CUDA instead, change the --index-url:
#   CUDA 12.1: https://download.pytorch.org/whl/cu121
#   CUDA 11.8: https://download.pytorch.org/whl/cu118
info "Installing PyTorch (CPU)..."
"$VENV_PIP" install --quiet \
    "torch>=2.2" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
success "PyTorch installed."

# Install everything else from requirements.txt.
# torch is already present so pip will skip it and install the remainder.
info "Installing remaining packages from requirements.txt..."
"$VENV_PIP" install --quiet -r "$REQ_FILE"
success "All packages installed."

# spacy-experimental: optional SRL fallback (AllenNLP dropped Python 3.11+ support)
"$VENV_PIP" install --quiet spacy-experimental 2>/dev/null && \
    success "spacy-experimental installed (optional SRL fallback)." || \
    warn "spacy-experimental skipped (optional — SRL will use dep-tree mapping)."

# ── Download spaCy language models ────────────────────────────────────────────
header "[3/7] Downloading spaCy models"

download_spacy_model() {
    local model="$1"
    if "$VENV_PYTHON" -c "import spacy; spacy.load('$model')" &>/dev/null 2>&1; then
        warn "$model already present — skipping."
    else
        info "Downloading $model ..."
        "$VENV_PYTHON" -m spacy download "$model" --quiet
        success "$model downloaded."
    fi
}

download_spacy_model "en_core_web_trf"   # Transformer English model (best accuracy)
download_spacy_model "en_core_web_sm"    # Small English (fast fallback)

# Vietnamese: underthesea bundles its own models; download them here
info "Downloading underthesea Vietnamese models..."
"$VENV_PYTHON" - <<'PYEOF'
try:
    from underthesea import sent_tokenize, word_tokenize, ner
    # Trigger model downloads
    sent_tokenize("Hợp đồng này được ký kết vào ngày 01 tháng 01 năm 2024.")
    word_tokenize("Hợp đồng mua bán hàng hoá.")
    print("  ✓ underthesea models ready")
except Exception as e:
    print(f"  ⚠ underthesea models: {e}")
PYEOF

# ── PhoBERT (Vietnamese BERT) — pre-download ──────────────────────────────────
header "[4/7] Pre-downloading PhoBERT model weights"
info "Downloading vinai/phobert-base-v2 (may be large)..."
"$VENV_PYTHON" - <<'PYEOF'
try:
    from transformers import AutoTokenizer, AutoModel
    AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    # Only download config/tokenizer — full weights downloaded on first use
    print("  ✓ PhoBERT tokenizer cached")
except Exception as e:
    print(f"  ⚠ PhoBERT cache: {e}")
PYEOF

# ── .env file (API keys) ──────────────────────────────────────────────────────
header "[5/7] Configuring environment variables"
if [ -f "$ENV_FILE" ]; then
    warn ".env already exists — not overwriting."
else
    cat > "$ENV_FILE" <<'ENVEOF'
# NLP Assignment — Environment Variables
# ----------------------------------------
# Google Gemini API Key (required for Assignment 3 RAG chatbot)
# Get yours at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# HuggingFace token (optional — only needed for gated models)
# Get yours at: https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN=

# ChromaDB persistence directory (default: data/chroma_db)
CHROMA_DB_PATH=data/chroma_db

# Embedding model for RAG vector store
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Top-k clauses to retrieve in RAG
RAG_TOP_K=5
ENVEOF
    success ".env created → $ENV_FILE"
    warn "IMPORTANT: Edit .env and replace 'your_gemini_api_key_here' with your actual key."
fi

# Prompt user to enter Gemini API key if running interactively
if [ -t 0 ]; then
    echo ""
    echo -e "${BOLD}Enter your Gemini API key (press Enter to skip and edit .env manually):${NC}"
    read -r -p "  GEMINI_API_KEY: " GEMINI_KEY
    if [ -n "$GEMINI_KEY" ]; then
        sed -i "s|your_gemini_api_key_here|$GEMINI_KEY|" "$ENV_FILE"
        success "Gemini API key saved to .env"
    else
        warn "Skipped — remember to set GEMINI_API_KEY in .env before running Assignment 3."
    fi
fi

# ── requirements.txt ──────────────────────────────────────────────────────────
header "[6/7] Generating requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    "$VENV_PIP" freeze --quiet > "$REQ_FILE"
    success "requirements.txt generated → $REQ_FILE"
else
    warn "requirements.txt already exists — skipping freeze."
fi

# ── Smoke test ────────────────────────────────────────────────────────────────
header "[7/7] Smoke test"
"$VENV_PYTHON" - <<'PYEOF'
results = []
tests = [
    ("spacy",                lambda: __import__("spacy")),
    ("transformers",         lambda: __import__("transformers")),
    ("datasets",             lambda: __import__("datasets")),
    ("sentence_transformers",lambda: __import__("sentence_transformers")),
    ("chromadb",             lambda: __import__("chromadb")),
    ("sklearn",              lambda: __import__("sklearn")),
    ("streamlit",            lambda: __import__("streamlit")),
    ("google.generativeai",  lambda: __import__("google.generativeai")),
    ("langchain",            lambda: __import__("langchain")),
    ("underthesea",          lambda: __import__("underthesea")),
    ("torch",                lambda: __import__("torch")),
]
for name, fn in tests:
    try:
        fn()
        results.append((name, True))
    except ImportError:
        results.append((name, False))

print("\n  Import check:")
for name, ok in results:
    status = "✓" if ok else "✗ MISSING"
    colour = "\033[0;32m" if ok else "\033[0;31m"
    print(f"  {colour}  {status:10}  {name}\033[0m")

failed = [n for n, ok in results if not ok]
if failed:
    print(f"\n  ⚠ Missing packages: {', '.join(failed)}")
    print("  Re-run this script or: pip install " + " ".join(failed))
else:
    print("\n  All packages OK!")
PYEOF

# ── Final instructions ────────────────────────────────────────────────────────
echo ""
header "=== Setup Complete ==="
echo ""
info "Activate the environment with:"
echo -e "    ${BOLD}source $VENV_DIR/bin/activate${NC}"
echo ""
info "Download datasets:"
echo -e "    ${BOLD}bash scripts/download_datasets.sh${NC}"
echo ""
info "Edit .env with your Gemini API key before running Assignment 3:"
echo -e "    ${BOLD}nano $ENV_FILE${NC}"
echo ""
