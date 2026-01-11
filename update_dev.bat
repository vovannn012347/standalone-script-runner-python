@echo off
echo.
echo === update execute.py in dist ===
echo.

set "TARGET=dist\executor\_internal\executor.py"
set "SOURCE=src\executor.py"

if not exist "%SOURCE%" (
    echo error: src\executor.py not found
    goto :error
)

if not exist "dist\executor\_internal" (
    echo error: directory _internal does nto exist
    echo run build_dev.bat first
    goto :error
)

copy /Y "%SOURCE%" "%TARGET%" >nul

if errorlevel 1 (
    echo Could not copy file
    goto :error
)

echo Complete. executor.py updated.
echo.

pause
exit /b 0

:error
echo.
echo error!
pause
exit /b 1