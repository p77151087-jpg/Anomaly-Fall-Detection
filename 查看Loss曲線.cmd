@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m fall_detect loss --port 0
if errorlevel 1 pause
