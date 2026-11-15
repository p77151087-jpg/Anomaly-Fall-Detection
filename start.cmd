@echo off
setlocal
rem Use Windows PowerShell modules when launched from PowerShell 7.
set "PSModulePath=%SystemRoot%\System32\WindowsPowerShell\v1.0\Modules"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" %*
set "FALL_DETECT_EXIT=%ERRORLEVEL%"
if not "%FALL_DETECT_EXIT%"=="0" echo Setup or launch failed. See the message above.
exit /b %FALL_DETECT_EXIT%
