@echo off
rem ============================================================
rem  Remove the logon autostart entry (does not stop a running
rem  server; use stop_server.bat for that).
rem  Cleans both the new VBS launcher and the old .lnk shortcut.
rem ============================================================
setlocal
set "DST=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\PPG-Platform.vbs"
set "LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\PPG-Platform.lnk"
set "FOUND="
if exist "%DST%" (
  del /f /q "%DST%"
  echo [OK] removed: %DST%
  set "FOUND=1"
)
if exist "%LNK%" (
  del /f /q "%LNK%"
  echo [OK] removed: %LNK%
  set "FOUND=1"
)
if not defined FOUND echo [INFO] autostart was not installed.
echo.
pause
