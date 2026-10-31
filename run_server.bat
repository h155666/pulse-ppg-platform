@echo off
rem ============================================================
rem  PPG platform - resident server launcher
rem  Usage:  run_server.bat [port]        (default 8502)
rem  Logs :  logs\server.log
rem
rem  Kept ASCII-only on purpose: cmd.exe reads .bat with the
rem  OEM/GBK code page, so UTF-8 Chinese text would be garbled.
rem  The Chinese project path is resolved by %%~dp0 at runtime.
rem ============================================================
setlocal
cd /d "%~dp0"
if not exist "logs" mkdir "logs"

set "PORT=8502"
if not "%~1"=="" set "PORT=%~1"

set "PY=C:\Users\15778\.workbuddy\binaries\python\envs\pulse\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] venv python not found:
  echo         %PY%
  echo Create the venv first, or edit PY inside this file.
  exit /b 1
)

echo.>> "logs\server.log"
echo [%date% %time%] starting streamlit on port %PORT%>> "logs\server.log"
"%PY%" -m streamlit run app.py --server.port %PORT% --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false --server.fileWatcherType auto >> "logs\server.log" 2>&1
