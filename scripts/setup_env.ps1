# =============================================================================
# setup_env.ps1 — NLP Assignment Environment Setup (Windows)
#
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# What this script does:
#   1. Checks for Python 3.12
#   2. Creates a virtual environment in .venv\
#   3. Installs all Python dependencies (requirements.txt)
#   4. Downloads spaCy English models
#   5. Pre-caches PhoBERT tokenizer from HuggingFace
#   6. Creates .env from .env.example if it does not exist
#   7. Runs a smoke-test import check
# =============================================================================

#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Info    { param($Msg) Write-Host "  [INFO]  $Msg" -ForegroundColor Cyan }
function Success { param($Msg) Write-Host "  [ OK ]  $Msg" -ForegroundColor Green }
function Warn    { param($Msg) Write-Host "  [WARN]  $Msg" -ForegroundColor Yellow }
function Err     { param($Msg) Write-Host "  [ERR ]  $Msg" -ForegroundColor Red }
function Header  { param($Msg) Write-Host "`n$Msg" -ForegroundColor Green }

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir     = Join-Path $ProjectRoot ".venv"
$VenvPython  = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip     = Join-Path $VenvDir "Scripts\pip.exe"
$EnvFile     = Join-Path $ProjectRoot ".env"
$EnvExample  = Join-Path $ProjectRoot ".env.example"
$ReqFile     = Join-Path $ProjectRoot "requirements.txt"

Set-Location $ProjectRoot

Header "=== NLP Assignment — Environment Setup (Windows) ==="
Info "Project root : $ProjectRoot"
Info "Virtualenv   : $VenvDir"

# ---------------------------------------------------------------------------
# Step 1 — Python version check
# ---------------------------------------------------------------------------
Header "[1/7] Checking Python"

$PythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.(1[0-9]|[2-9]\d)") {
            $PythonCmd = $cmd
            break
        }
    } catch { }
}

if (-not $PythonCmd) {
    Err "Python 3.10+ not found. Download from https://www.python.org/downloads/"
    Err "Ensure 'Add Python to PATH' is checked during installation."
    exit 1
}

$PythonVersion = & $PythonCmd --version 2>&1
Info "Using: $PythonCmd ($PythonVersion)"

# ---------------------------------------------------------------------------
# Step 2 — Create virtual environment
# ---------------------------------------------------------------------------
Header "[2/7] Creating virtual environment"

if (Test-Path $VenvPython) {
    Warn "Virtualenv already exists at $VenvDir — skipping creation."
} else {
    & $PythonCmd -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Err "Failed to create virtual environment."
        exit 1
    }
    Success "Virtualenv created → $VenvDir"
}

# Upgrade pip silently
Info "Upgrading pip / setuptools / wheel..."
& $VenvPython -m pip install --quiet --upgrade pip setuptools wheel
Success "pip upgraded."

# ---------------------------------------------------------------------------
# Step 3 — Install Python packages
# ---------------------------------------------------------------------------
Header "[3/7] Installing Python packages"
Info "This may take 5–15 minutes on first run..."

# Install PyTorch first via the official Windows wheel index (CPU build).
# requirements.txt lists torch>=2.2; pip treats it as satisfied once torch is
# already present, so the remainder of requirements.txt installs cleanly.
# To use CUDA instead, change the --index-url:
#   CUDA 12.1: https://download.pytorch.org/whl/cu121
#   CUDA 11.8: https://download.pytorch.org/whl/cu118
Info "Installing PyTorch (CPU build for Windows)..."
& $VenvPip install --quiet `
    "torch>=2.2" torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) {
    Err "PyTorch install failed. Check your internet connection and retry."
    exit 1
}
Success "PyTorch installed."

# Install everything else from requirements.txt.
# torch is already present so pip will skip it and install the remainder.
Info "Installing remaining packages from requirements.txt..."
& $VenvPip install --quiet -r $ReqFile
if ($LASTEXITCODE -ne 0) {
    Err "Package installation failed. Run manually: pip install -r requirements.txt"
    exit 1
}
Success "All packages installed."

# spacy-experimental: optional SRL fallback (AllenNLP dropped Python 3.11+ support)
Info "Installing spacy-experimental (optional SRL fallback)..."
& $VenvPip install --quiet spacy-experimental 2>$null
if ($LASTEXITCODE -eq 0) {
    Success "spacy-experimental installed."
} else {
    Warn "spacy-experimental skipped (optional — SRL will use dep-tree mapping)."
}

Success "All packages installed."

# ---------------------------------------------------------------------------
# Step 4 — Download spaCy models
# ---------------------------------------------------------------------------
Header "[4/7] Downloading spaCy language models"

function Get-SpacyModel {
    param($Model)
    $check = & $VenvPython -c "import spacy; spacy.load('$Model')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Warn "$Model already present — skipping."
    } else {
        Info "Downloading $Model ..."
        & $VenvPython -m spacy download $Model --quiet
        if ($LASTEXITCODE -eq 0) {
            Success "$Model downloaded."
        } else {
            Warn "$Model download failed — you can retry manually: python -m spacy download $Model"
        }
    }
}

Get-SpacyModel "en_core_web_trf"
Get-SpacyModel "en_core_web_sm"

# ---------------------------------------------------------------------------
# Step 5 — Pre-cache PhoBERT tokenizer
# ---------------------------------------------------------------------------
Header "[5/7] Pre-caching PhoBERT tokenizer"

$PhobertScript = @"
try:
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
    print('  PhoBERT tokenizer cached.')
except Exception as e:
    print(f'  Warning: PhoBERT cache failed: {e}')
"@
& $VenvPython -c $PhobertScript

# ---------------------------------------------------------------------------
# Step 6 — Create .env file
# ---------------------------------------------------------------------------
Header "[6/7] Configuring environment variables"

if (Test-Path $EnvFile) {
    Warn ".env already exists — not overwriting."
} else {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
        Success ".env created from .env.example → $EnvFile"
    } else {
        @"
# NLP Assignment — Environment Variables
# ----------------------------------------
# Google Gemini API Key (required for Assignment 3 RAG chatbot)
# Get yours at: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

GEMINI_MODEL=gemini-1.5-flash
CHROMA_PERSIST_DIR=./chroma_db
TOP_K_RESULTS=3
"@ | Set-Content -Path $EnvFile -Encoding UTF8
        Success ".env created → $EnvFile"
    }
    Warn "IMPORTANT: Edit .env and replace 'your_google_api_key_here' with your actual key."
    Warn "           Open with: notepad $EnvFile"
}

# Optional: prompt for key interactively
Write-Host ""
Write-Host "  Enter your Google API key (or press Enter to skip and edit .env manually):" -ForegroundColor Cyan
$GeminiKey = Read-Host "  GOOGLE_API_KEY"
if ($GeminiKey -and $GeminiKey.Trim() -ne "") {
    (Get-Content $EnvFile -Raw) -replace 'your_google_api_key_here', $GeminiKey.Trim() |
        Set-Content -Path $EnvFile -Encoding UTF8 -NoNewline
    Success "Google API key saved to .env"
} else {
    Warn "Skipped — remember to set GOOGLE_API_KEY in .env before running Assignment 3."
}

# ---------------------------------------------------------------------------
# Step 7 — Smoke test
# ---------------------------------------------------------------------------
Header "[7/7] Smoke test — import check"

$SmokeScript = @"
import sys
packages = [
    ("spacy",                 "spacy"),
    ("transformers",          "transformers"),
    ("datasets",              "datasets"),
    ("sentence_transformers", "sentence_transformers"),
    ("chromadb",              "chromadb"),
    ("sklearn",               "sklearn"),
    ("streamlit",             "streamlit"),
    ("google.generativeai",   "google.generativeai"),
    ("langchain",             "langchain"),
    ("torch",                 "torch"),
    ("dotenv",                "dotenv"),
    ("tqdm",                  "tqdm"),
    ("kaggle",                "kaggle"),
]
results = []
for display, imp in packages:
    try:
        __import__(imp)
        results.append((display, True))
    except ImportError:
        results.append((display, False))

print("\n  Import check:")
for name, ok in results:
    icon = "OK " if ok else "FAIL"
    print(f"  [{icon}]  {name}")

failed = [n for n, ok in results if not ok]
if failed:
    print(f"\n  Missing: {', '.join(failed)}")
    print(f"  Re-run: pip install {' '.join(failed)}")
    sys.exit(1)
else:
    print("\n  All packages OK!")
"@

& $VenvPython -c $SmokeScript
if ($LASTEXITCODE -ne 0) {
    Warn "Some packages failed to import. See above for details."
} else {
    Success "Smoke test passed."
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Header "=== Setup Complete ==="
Write-Host ""
Info "Activate the environment with:"
Write-Host "    .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "    # or for CMD:"
Write-Host "    .venv\Scripts\activate.bat" -ForegroundColor White
Write-Host ""
Info "Download datasets next:"
Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1" -ForegroundColor White
Write-Host ""
Info "Edit .env with your Google API key (needed for Assignment 3):"
Write-Host "    notepad .env" -ForegroundColor White
Write-Host ""
