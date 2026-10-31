@echo off
rem ============================================================
rem  Optional: make the server start automatically at logon.
rem  Creates a shortcut in the user's Startup folder.
rem  No administrator rights needed. Run uninstall_autostart.bat
rem  to undo. Double-click once.
rem ============================================================
setlocal
set "LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\PPG-Platform.lnk"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%LNK%'); $s.TargetPath='%~dp0run_server.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.Description='PPG pulse wave platform (resident)'; $s.Save()"
if exist "%LNK%" (
  echo [OK] autostart installed:
  echo      %LNK%
) else (
  echo [FAIL] could not create the shortcut.
)
echo.
echo The server will start (minimized) at next logon.
timeout /t 4 >nul
