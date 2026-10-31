@echo off
rem ============================================================
rem  Optional: make the server start automatically at logon.
rem  Copies the hidden-window VBS launcher into the user's
rem  Startup folder. No administrator rights needed.
rem  Run uninstall_autostart.bat to undo. Double-click once.
rem
rem  Why a .vbs instead of a .lnk:  a shortcut still shows a
rem  minimized console window at logon, while the VBS runs
rem  run_server.bat fully hidden (window style 0). The template
rem  is pure ASCII so cmd/WScript cannot garble the Chinese path.
rem ============================================================
setlocal
set "DST=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\PPG-Platform.vbs"
if not exist "%~dp0tools\autostart_template.vbs" (
  echo [FAIL] tools\autostart_template.vbs not found.
  exit /b 1
)
copy /y "%~dp0tools\autostart_template.vbs" "%DST%" >nul
if exist "%DST%" (
  echo [OK] autostart installed:
  echo      %DST%
  echo The server will start hidden at next logon.
  echo It is already running now if you started it with run_server.bat.
) else (
  echo [FAIL] could not copy the autostart file.
)
echo.
pause
