@echo off
setlocal
where pythonw.exe >nul 2>&1
if not errorlevel 1 (
    start "NPCMicro13M GUI" pythonw.exe "%~dp0uomind_gui.pyw"
    exit /b 0
)
where pyw.exe >nul 2>&1
if not errorlevel 1 (
    start "NPCMicro13M GUI" pyw.exe "%~dp0uomind_gui.pyw"
    exit /b 0
)
for /d %%D in ("%LocalAppData%\Programs\Python\Python*") do (
    if exist "%%~fD\pythonw.exe" (
        start "NPCMicro13M GUI" "%%~fD\pythonw.exe" "%~dp0uomind_gui.pyw"
        exit /b 0
    )
)
echo Python was not found on PATH.
echo Double-click uomind_gui.pyw after associating it with pythonw.exe,
echo or install Python and ensure its launcher is available.
pause
