@echo off
rem ============================================================
rem  Remove the logon autostart shortcut (does not stop a
rem  running server; use stop_server.bat for that).
rem ============================================================
setlocal
set "LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\PPG-Platform.lnk"
if exist "%LNK%" (
  del /f /q "%LNK%"
  echo [OK] autostart removed.
) else (
  echo [INFO] autostart was not installed.
)
timeout /t 3 >nul
