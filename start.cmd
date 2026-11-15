@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" %*
set "FALL_DETECT_EXIT=%ERRORLEVEL%"
if not "%FALL_DETECT_EXIT%"=="0" echo Setup or launch failed. See the message above.
exit /b %FALL_DETECT_EXIT%
