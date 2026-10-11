# Kian launcher for Windows (PowerShell). Safe: it never changes system/security
# settings. It sets up a local Python environment, gives birth to Kian if needed,
# and opens the living face in your browser.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$py = "python"
if (-not (Get-Command $py -ErrorAction SilentlyContinue)) { $py = "py" }

if (-not (Test-Path ".venv")) {
  Write-Host "Creating virtual environment..." -ForegroundColor Cyan
  & $py -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
& ".venv\Scripts\python.exe" -m pip install --quiet numpy
# Optional senses (screen/mouse/web). Comment out to keep it minimal.
& ".venv\Scripts\python.exe" -m pip install --quiet pillow mss pyautogui requests 2>$null

if (-not (Test-Path "kian_home\being.json")) {
  Write-Host "Giving birth to Kian..." -ForegroundColor Green
  & ".venv\Scripts\python.exe" -m kian --home kian_home birth
}
Write-Host "Launching Kian's face..." -ForegroundColor Green
& ".venv\Scripts\python.exe" -m kian --home kian_home live --port 7860
