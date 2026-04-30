#!/usr/bin/env bash
# =============================================================================
# NLP Assignment — Dataset Downloader
# Downloads ALL datasets for both English and Vietnamese pipelines:
#   [EN-1] Atticus Open Contract Dataset (Kaggle)   ← ACTIVE
#   [EN-2] Business Contract Dataset (GitHub)        ← commented out
#   [VI-1] Vietnamese Legal Documents — th1nhng0 (HuggingFace)  ← commented out
#   [VI-2] Vietnamese Legal Documents — YuITC (HuggingFace)     ← commented out
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}  ℹ  $*${NC}"; }
success() { echo -e "${GREEN}  ✓  $*${NC}"; }
warn()    { echo -e "${YELLOW}  ⚠  $*${NC}"; }
error()   { echo -e "${RED}  ✗  $*${NC}"; }
header()  { echo -e "\n${BOLD}${GREEN}$*${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
header "=== NLP Assignment — Dataset Downloader ==="
info "Project root : $PROJECT_ROOT"
info "Data dir     : $DATA_DIR"

# Verify Python 3 is available
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Please install Python 3.9+."
    exit 1
fi
PYTHON="$(command -v python3)"
info "Python       : $PYTHON ($($PYTHON --version 2>&1))"

# Create target directories
mkdir -p \
    "$DATA_DIR/english/atticus" \
    "$DATA_DIR/contracts/english"
#   "$DATA_DIR/english/github" \
#   "$DATA_DIR/vietnamese/th1nhng0" \
#   "$DATA_DIR/vietnamese/yuiteam" \
#   "$DATA_DIR/contracts/vietnamese"

# ── [EN-1] Atticus Contract Dataset from Kaggle ───────────────────────────────
header "[EN-1] Atticus Open Contract Dataset (Kaggle)"
ATTICUS_DIR="$DATA_DIR/english/atticus"

if [ -n "$(ls -A "$ATTICUS_DIR" 2>/dev/null)" ]; then
    warn "Atticus data already exists in $ATTICUS_DIR — skipping."
else
    # Ensure kaggle CLI is installed
    if ! $PYTHON -c "import kaggle" &>/dev/null 2>&1; then
        info "Installing kaggle CLI..."
        $PYTHON -m pip install -q kaggle
    fi

    # Check for Kaggle credentials
    KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
    if [ ! -f "$KAGGLE_JSON" ]; then
        warn "Kaggle credentials not found at ~/.kaggle/kaggle.json"
        echo ""
        echo "  To set up Kaggle credentials:"
        echo "  1. Go to https://www.kaggle.com/settings → Account → API"
        echo "  2. Click 'Create New Token' — downloads kaggle.json"
        echo "  3. Run:"
        echo "       mkdir -p ~/.kaggle"
        echo "       mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json"
        echo "       chmod 600 ~/.kaggle/kaggle.json"
        echo ""
        warn "Skipping Kaggle download. Re-run this script after setting up credentials."
    else
        chmod 600 "$KAGGLE_JSON"
        info "Downloading konradb/atticus-open-contract-dataset-aok-beta..."
        $PYTHON -m kaggle datasets download \
            -d konradb/atticus-open-contract-dataset-aok-beta \
            -p "$ATTICUS_DIR" \
            --unzip \
            --quiet
        success "Atticus dataset → $ATTICUS_DIR"

        # Copy a sample of contract text files to contracts/english for pipeline input
        info "Copying sample contracts to data/contracts/english/ ..."
        find "$ATTICUS_DIR" -name "*.txt" | head -20 | \
            xargs -I{} cp {} "$DATA_DIR/contracts/english/" 2>/dev/null || true
        find "$ATTICUS_DIR" -name "*.csv" | head -5 | \
            xargs -I{} cp {} "$DATA_DIR/contracts/english/" 2>/dev/null || true
        success "Sample contracts copied."
    fi
fi

# ── [EN-2] Business Contract Dataset from GitHub ─────────────────────────────
# header "[EN-2] Business Contract Dataset (GitHub)"
# GITHUB_DIR="$DATA_DIR/english/github"
#
# if [ -d "$GITHUB_DIR/.git" ]; then
#     info "Repository already cloned. Pulling latest..."
#     git -C "$GITHUB_DIR" pull --quiet
#     success "GitHub dataset up to date → $GITHUB_DIR"
# else
#     info "Cloning meghaarajeev/Business-Contract-Dataset..."
#     git clone --quiet \
#         https://github.com/meghaarajeev/Business-Contract-Dataset-Intel-Training--Program-2024 \
#         "$GITHUB_DIR"
#     success "GitHub English dataset → $GITHUB_DIR"
#
#     # Copy .txt/.pdf/.csv contract files to contracts/english
#     find "$GITHUB_DIR" -name "*.txt" -o -name "*.csv" | head -20 | \
#         xargs -I{} cp {} "$DATA_DIR/contracts/english/" 2>/dev/null || true
# fi

# ── [VI-1 & VI-2] Vietnamese Legal Documents (HuggingFace) ───────────────────
# header "[VI-1 & VI-2] Vietnamese Legal Documents (HuggingFace)"
#
# # Ensure 'datasets' library is installed
# if ! $PYTHON -c "import datasets" &>/dev/null 2>&1; then
#     info "Installing HuggingFace 'datasets' library..."
#     $PYTHON -m pip install -q datasets huggingface-hub
# fi
#
# export DATA_DIR="$DATA_DIR"
# $PYTHON - <<PYEOF
# import os, sys
# from pathlib import Path
#
# data_dir = Path("$DATA_DIR")
# vi_th1_dir = data_dir / "vietnamese" / "th1nhng0"
# vi_yu_dir  = data_dir / "vietnamese" / "yuiteam"
# contracts_vi = data_dir / "contracts" / "vietnamese"
#
# def download_hf(repo_id, save_path, description):
#     from datasets import load_dataset
#     print(f"  Downloading {repo_id}...")
#     try:
#         ds = load_dataset(repo_id, trust_remote_code=True)
#         ds.save_to_disk(str(save_path))
#         print(f"  ✓ Saved to {save_path}")
#
#         split = list(ds.keys())[0]
#         sample_df = ds[split].to_pandas()
#         text_col = next((c for c in sample_df.columns
#                          if any(k in c.lower() for k in ['text','content','body','document','van_ban'])),
#                         sample_df.columns[0])
#         contracts_vi.mkdir(parents=True, exist_ok=True)
#         for i, row in sample_df.head(30).iterrows():
#             fname = contracts_vi / f"{description}_{i:04d}.txt"
#             fname.write_text(str(row[text_col]), encoding='utf-8')
#         print(f"  ✓ Exported 30 samples to {contracts_vi}/")
#         return True
#     except Exception as e:
#         print(f"  Warning [{repo_id}]: {e}")
#         return False
#
# skip_th1 = vi_th1_dir.exists() and any(vi_th1_dir.iterdir())
# skip_yu  = vi_yu_dir.exists()  and any(vi_yu_dir.iterdir())
#
# if skip_th1:
#     print("  ✓ th1nhng0 already downloaded — skipping")
# else:
#     download_hf("th1nhng0/vietnamese-legal-documents", vi_th1_dir, "th1nhng0")
#
# if skip_yu:
#     print("  ✓ YuITC already downloaded — skipping")
# else:
#     download_hf("YuITC/Vietnamese-Legal-Documents", vi_yu_dir, "yuiteam")
# PYEOF

# ── Summary ───────────────────────────────────────────────────────────────────
header "=== Download Summary ==="
echo ""
printf "  %-45s  %s\n" "Location" "Status"
printf "  %-45s  %s\n" "--------" "------"

check_dir() {
    local path="$1"; local label="$2"
    if [ -d "$path" ] && [ -n "$(ls -A "$path" 2>/dev/null)" ]; then
        local count; count=$(find "$path" -maxdepth 2 -type f | wc -l)
        printf "  ${GREEN}%-45s  ✓ %d file(s)${NC}\n" "$label" "$count"
    else
        printf "  ${YELLOW}%-45s  ⚠ empty / skipped${NC}\n" "$label"
    fi
}

check_dir "$DATA_DIR/english/atticus"          "data/english/atticus       [EN-1 Kaggle]"
# check_dir "$DATA_DIR/english/github"           "data/english/github        [EN-2 GitHub]"
# check_dir "$DATA_DIR/vietnamese/th1nhng0"      "data/vietnamese/th1nhng0   [VI-1 HF]"
# check_dir "$DATA_DIR/vietnamese/yuiteam"       "data/vietnamese/yuiteam    [VI-2 HF]"
check_dir "$DATA_DIR/contracts/english"        "data/contracts/english     [Pipeline input]"
# check_dir "$DATA_DIR/contracts/vietnamese"     "data/contracts/vietnamese  [Pipeline input]"

echo ""
info "Next step: run  bash scripts/setup_env.sh  to install Python dependencies."
echo ""
