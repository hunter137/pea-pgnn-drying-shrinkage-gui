@echo off
setlocal
cd /d "%~dp0"
where pythonw.exe >nul 2>nul
if errorlevel 1 (
    python run_gui.py
) else (
    start "" pythonw.exe run_gui.py
)
endlocal
