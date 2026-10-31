@echo off
rem ============================================================
rem  Launch the PPG platform with the pre-built venv (Python 3.13).
rem  No system Python pollution. Double-click this file to start.
rem  The Chinese path is resolved at runtime via %%~dp0, so the
rem  file itself stays pure ASCII and never garbles.
rem ============================================================
cd /d "%~dp0"
"C:\Users\15778\.workbuddy\binaries\python\envs\pulse\Scripts\python.exe" -m streamlit run app.py --server.port 8502
pause
