# Register a Windows Scheduled Task so Kian evolves once a day automatically.
# This only creates a normal user task that runs: python -m kian evolve
# It changes NO security settings.
$ErrorActionPreference = "Stop"
$root = (Split-Path $PSScriptRoot -Parent)
$pyExe = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $pyExe)) { $pyExe = "python" }
$action  = New-ScheduledTaskAction -Execute $pyExe -Argument "-m kian --home `"$root\kian_home`" evolve" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At 3am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName "Kian Daily Evolution" -Action $action -Trigger $trigger -Settings $settings -Force
Write-Host "Scheduled: Kian will learn a little every day at 3:00 AM." -ForegroundColor Green
