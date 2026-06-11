$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Start-Process -FilePath ".\.venv\Scripts\pythonw.exe" -ArgumentList "run.py" -WorkingDirectory $PSScriptRoot
