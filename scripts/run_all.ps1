# =============================================================================
# run_all.ps1 — End-to-end NLP Assignment Pipeline (Windows)
#
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
#
# What this script does:
#   1. Creates .venv if it does not exist and installs dependencies
#   2. Downloads spaCy model en_core_web_sm if missing
#   3. Creates output\ and results\ directories
#   4. Runs Assignment 1 (clause splitting, NP chunking, dependency parsing)
#   5. Copies Assignment 1 outputs to results\assignment1\
#   6. Runs Assignment 2 (NER, SRL, intent classification)
#   7. Copies Assignment 2 outputs to results\assignment2\
#   8. Prints a final summary table
#
# Note: Assignment 3 (RAG chatbot) is not started automatically — run it
#       manually with:  streamlit run src\assignment3\app.py
# =============================================================================

#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Info    { param($Msg) Write-Host "  [INFO]   $Msg" -ForegroundColor Cyan }
function Success { param($Msg) Write-Host "  [ OK ]   $Msg" -ForegroundColor Green }
function Warn    { param($Msg) Write-Host "  [WARN]   $Msg" -ForegroundColor Yellow }
function Err     { param($Msg) Write-Host "  [ERROR]  $Msg" -ForegroundColor Red; exit 1 }
function Header  { param($Msg) Write-Host "`n============================================================" -ForegroundColor Cyan
                               Write-Host "  $Msg" -ForegroundColor Cyan
                               Write-Host "============================================================" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir     = Join-Path $ProjectRoot ".venv"
$VenvPython  = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip     = Join-Path $VenvDir "Scripts\pip.exe"
$ReqFile     = Join-Path $ProjectRoot "requirements.txt"
$OutputDir   = Join-Path $ProjectRoot "output"
$ResultsDir  = Join-Path $ProjectRoot "results"

Set-Location $ProjectRoot
Info "Project root: $ProjectRoot"

# ---------------------------------------------------------------------------
# Step 1 — Virtual environment
# ---------------------------------------------------------------------------
if (-not (Test-Path $VenvPython)) {
    Info "Creating virtual environment at .venv\ ..."

    $PythonCmd = $null
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $ver = & $cmd --version 2>&1
            if ($ver -match "Python 3\.(1[0-9]|[2-9]\d)") { $PythonCmd = $cmd; break }
        } catch { }
    }
    if (-not $PythonCmd) { Err "Python 3.10+ not found. Install from https://www.python.org/downloads/" }

    & $PythonCmd -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Err "Failed to create virtual environment." }
    Success "Virtual environment created."
} else {
    Info "Using existing virtual environment at .venv\"
}

# ---------------------------------------------------------------------------
# Step 2 — Install / verify dependencies
# ---------------------------------------------------------------------------
Info "Installing / verifying Python dependencies from requirements.txt ..."
& $VenvPython -m pip install --quiet --upgrade pip 2>$null

# PyTorch via Windows wheel index first (so requirements.txt skips it cleanly)
& $VenvPip install --quiet "torch>=2.2" torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cpu 2>$null

& $VenvPip install --quiet -r $ReqFile
if ($LASTEXITCODE -ne 0) { Err "Dependency install failed. Run setup_env.ps1 first." }
Success "Dependencies OK."

# ---------------------------------------------------------------------------
# Step 3 — spaCy model
# ---------------------------------------------------------------------------
$modelCheck = & $VenvPython -c "import spacy; spacy.load('en_core_web_sm')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Info "Downloading spaCy model en_core_web_sm ..."
    & $VenvPython -m spacy download en_core_web_sm --quiet
    if ($LASTEXITCODE -ne 0) { Err "spaCy model download failed." }
    Success "spaCy model downloaded."
} else {
    Info "spaCy model en_core_web_sm already present."
}

# ---------------------------------------------------------------------------
# Step 4 — Create directories
# ---------------------------------------------------------------------------
foreach ($dir in @(
    $OutputDir,
    (Join-Path $ResultsDir "assignment1"),
    (Join-Path $ResultsDir "assignment2")
)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# ---------------------------------------------------------------------------
# Step 5 — Assignment 1
# ---------------------------------------------------------------------------
Header "ASSIGNMENT 1 — Preprocessing & Syntax Analysis"
Info "Running Assignment 1 ..."

& $VenvPython (Join-Path $ProjectRoot "src\assignment1\run_assignment1.py")
if ($LASTEXITCODE -ne 0) { Err "Assignment 1 failed. See output above." }
Success "Assignment 1 complete."

Info "Copying Assignment 1 results to results\assignment1\ ..."
foreach ($f in @("clauses.txt", "chunks.txt", "dependency.json")) {
    $src = Join-Path $OutputDir $f
    $dst = Join-Path $ResultsDir "assignment1\$f"
    if (Test-Path $src) { Copy-Item -Path $src -Destination $dst -Force }
}

# ---------------------------------------------------------------------------
# Step 6 — Assignment 2
# ---------------------------------------------------------------------------
Header "ASSIGNMENT 2 — Information Extraction & Semantic Analysis"
Info "Running Assignment 2 ..."

& $VenvPython (Join-Path $ProjectRoot "src\assignment2\run_assignment2.py")
if ($LASTEXITCODE -ne 0) { Err "Assignment 2 failed. See output above." }
Success "Assignment 2 complete."

Info "Copying Assignment 2 results to results\assignment2\ ..."
foreach ($f in @("ner_results.json", "srl_results.json", "intent_classification.txt")) {
    $src = Join-Path $OutputDir $f
    $dst = Join-Path $ResultsDir "assignment2\$f"
    if (Test-Path $src) { Copy-Item -Path $src -Destination $dst -Force }
}

# ---------------------------------------------------------------------------
# Step 7 — Summary
# ---------------------------------------------------------------------------
Header "PIPELINE COMPLETE"
Write-Host ""

function Show-FileStatus {
    param($RelPath, $Label)
    $full = Join-Path $ProjectRoot $RelPath
    if (Test-Path $full) {
        $kb = [math]::Round((Get-Item $full).Length / 1KB, 1)
        Write-Host ("  {0,-50}  " -f $Label) -NoNewline
        Write-Host ("{0,6} KB" -f $kb) -ForegroundColor Green
    } else {
        Write-Host ("  {0,-50}  " -f $Label) -NoNewline
        Write-Host "MISSING" -ForegroundColor Red
    }
}

Write-Host "  Spec-required outputs (output\)" -ForegroundColor White
Show-FileStatus "output\clauses.txt"               "output\clauses.txt"
Show-FileStatus "output\chunks.txt"                "output\chunks.txt"
Show-FileStatus "output\dependency.json"           "output\dependency.json"
Show-FileStatus "output\ner_results.json"          "output\ner_results.json"
Show-FileStatus "output\srl_results.json"          "output\srl_results.json"
Show-FileStatus "output\intent_classification.txt" "output\intent_classification.txt"

Write-Host ""
Write-Host "  Organised copies (results\)" -ForegroundColor White
Show-FileStatus "results\assignment1\clauses.txt"               "results\assignment1\clauses.txt"
Show-FileStatus "results\assignment1\chunks.txt"                "results\assignment1\chunks.txt"
Show-FileStatus "results\assignment1\dependency.json"           "results\assignment1\dependency.json"
Show-FileStatus "results\assignment2\ner_results.json"          "results\assignment2\ner_results.json"
Show-FileStatus "results\assignment2\srl_results.json"          "results\assignment2\srl_results.json"
Show-FileStatus "results\assignment2\intent_classification.txt" "results\assignment2\intent_classification.txt"

Write-Host ""
Write-Host "  Assignment 3 (RAG chatbot) — run manually:" -ForegroundColor Cyan
Write-Host "    streamlit run src\assignment3\app.py" -ForegroundColor White
Write-Host ""
Success "All done!"
