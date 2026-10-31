' ============================================================
'  PPG platform - resident autostart template (hidden window)
'  install_autostart.bat copies this file to the user's Startup
'  folder, so the Streamlit server starts invisibly at logon.
'
'  Pure ASCII on purpose: WScript reads .vbs with the ANSI (GBK)
'  code page, so UTF-8 Chinese text would be garbled. The Chinese
'  folder name is built from Unicode code points instead:
'  U+9F99=40857 (long)  U+867E=34430 (xia)
'
'  NOTE: if you move the project, re-run install_autostart.bat
'  (or edit the path below).
' ============================================================
Dim sh, p
p = "C:\Users\15778\Desktop\" & ChrW(40857) & ChrW(34430) & _
    "\pulse_ppg_streamlit\run_server.bat"
Set sh = CreateObject("WScript.Shell")
sh.Run """" & p & """", 0, False
