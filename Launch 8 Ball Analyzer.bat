@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
)
echo Installing/updating dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo Starting 8 Ball Pool Shot Analyzer...
start "" ".venv\Scripts\pythonw.exe" "run.py"
