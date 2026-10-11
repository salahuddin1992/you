@echo off
REM Kian launcher for Windows (CMD). Safe: no system/security changes.
cd /d "%~dp0\.."
where python >nul 2>nul && (set PY=python) || (set PY=py)
if not exist ".venv" ( echo Creating venv... & %PY% -m venv .venv )
call ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
call ".venv\Scripts\python.exe" -m pip install --quiet numpy
if not exist "kian_home\being.json" (
  echo Giving birth to Kian...
  call ".venv\Scripts\python.exe" -m kian --home kian_home birth
)
echo Launching Kian...
call ".venv\Scripts\python.exe" -m kian --home kian_home live --port 7860
