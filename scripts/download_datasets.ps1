# =============================================================================
# download_datasets.ps1 — NLP Assignment Dataset Downloader (Windows)
#
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1
#
# Downloads:
#   [EN-1] Atticus Open Contract Dataset (Kaggle)
#          → data\english\atticus\
#          → data\contracts\english\   (sample contract .txt files)
#
# Prerequisites:
#   - Kaggle API credentials at %USERPROFILE%\.kaggle\kaggle.json
#     Get them at: https://www.kaggle.com/settings → Account → API → Create New Token
#   - Python 3.10+ with the 'kaggle' package installed
#     (run scripts\setup_env.ps1 first, or: pip install kaggle)
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
$ProjectRoot    = Split-Path -Parent $PSScriptRoot
$DataDir        = Join-Path $ProjectRoot "data"
$AtticusDir     = Join-Path $DataDir "english\atticus"
$ContractsDir   = Join-Path $DataDir "contracts\english"
$KaggleJson     = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"

# Prefer the project venv Python; fall back to system Python
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
}

Set-Location $ProjectRoot

Header "=== NLP Assignment — Dataset Downloader (Windows) ==="
Info "Project root  : $ProjectRoot"
Info "Data dir      : $DataDir"
Info "Python        : $VenvPython"

# ---------------------------------------------------------------------------
# Create target directories
# ---------------------------------------------------------------------------
foreach ($dir in @($AtticusDir, $ContractsDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Info "Created directory: $dir"
    }
}

# ---------------------------------------------------------------------------
# [EN-1] Atticus Open Contract Dataset from Kaggle
# ---------------------------------------------------------------------------
Header "[EN-1] Atticus Open Contract Dataset (Kaggle)"

$atticusFiles = Get-ChildItem -Path $AtticusDir -ErrorAction SilentlyContinue
if ($atticusFiles.Count -gt 0) {
    Warn "Atticus data already exists in $AtticusDir ($($atticusFiles.Count) file(s)) — skipping."
} else {
    # Check Kaggle credentials
    if (-not (Test-Path $KaggleJson)) {
        Warn "Kaggle credentials not found at: $KaggleJson"
        Write-Host ""
        Write-Host "  To set up Kaggle credentials:" -ForegroundColor Yellow
        Write-Host "  1. Go to https://www.kaggle.com/settings → Account → API" -ForegroundColor Yellow
        Write-Host "  2. Click 'Create New Token' — this downloads kaggle.json" -ForegroundColor Yellow
        Write-Host "  3. Move the file to:" -ForegroundColor Yellow
        Write-Host "       $KaggleJson" -ForegroundColor White
        Write-Host ""
        Write-Host "  Then re-run this script." -ForegroundColor Yellow
        exit 1
    }

    # Ensure kaggle Python package is available
    $kaggleCheck = & $VenvPython -c "import kaggle" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Info "Installing Kaggle CLI package..."
        & $VenvPython -m pip install --quiet kaggle
        if ($LASTEXITCODE -ne 0) {
            Err "Failed to install kaggle package. Run: pip install kaggle"
            exit 1
        }
    }

    Info "Downloading konradb/atticus-open-contract-dataset-aok-beta..."
    & $VenvPython -m kaggle datasets download `
        -d konradb/atticus-open-contract-dataset-aok-beta `
        -p $AtticusDir `
        --unzip `
        --quiet

    if ($LASTEXITCODE -ne 0) {
        Err "Kaggle download failed. Check your credentials and internet connection."
        exit 1
    }
    Success "Atticus dataset → $AtticusDir"

    # Copy up to 20 .txt contract files to the pipeline input directory
    Info "Copying sample contracts to data\contracts\english\ ..."
    $txtFiles = Get-ChildItem -Path $AtticusDir -Filter "*.txt" -Recurse | Select-Object -First 20
    $csvFiles = Get-ChildItem -Path $AtticusDir -Filter "*.csv" -Recurse | Select-Object -First 5
    $copied = 0
    foreach ($f in ($txtFiles + $csvFiles)) {
        Copy-Item -Path $f.FullName -Destination $ContractsDir -ErrorAction SilentlyContinue
        $copied++
    }
    Success "Copied $copied sample contract files → $ContractsDir"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Header "=== Download Summary ==="
Write-Host ""

function Show-DirStatus {
    param($Path, $Label)
    if (Test-Path $Path) {
        $count = (Get-ChildItem -Path $Path -Recurse -File -ErrorAction SilentlyContinue).Count
        if ($count -gt 0) {
            Write-Host ("  {0,-50}  " -f $Label) -NoNewline
            Write-Host "OK ($count file(s))" -ForegroundColor Green
        } else {
            Write-Host ("  {0,-50}  " -f $Label) -NoNewline
            Write-Host "empty / skipped" -ForegroundColor Yellow
        }
    } else {
        Write-Host ("  {0,-50}  " -f $Label) -NoNewline
        Write-Host "not found" -ForegroundColor Yellow
    }
}

Show-DirStatus $AtticusDir   "data\english\atticus       [EN-1 Kaggle]"
Show-DirStatus $ContractsDir "data\contracts\english     [Pipeline input]"

Write-Host ""
Info "Next step: place your contract text in input\raw_contracts.txt"
Info "Then run:  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1"
Write-Host ""
