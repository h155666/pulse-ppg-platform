@echo off
rem ============================================================
rem  Stop the resident PPG server (kills the PID that LISTENs
rem  on the given port).  Usage: stop_server.bat [port]
rem ============================================================
setlocal enabledelayedexpansion
set "PORT=8502"
if not "%~1"=="" set "PORT=%~1"

set "FOUND="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do (
  if not "%%p"=="0" (
    echo Killing PID %%p on port %PORT% ...
    taskkill /F /PID %%p >nul 2>&1
    set "FOUND=1"
  )
)
if defined FOUND (
  echo [OK] port %PORT% released.
) else (
  echo [INFO] nothing is listening on port %PORT%.
)
timeout /t 2 >nul
