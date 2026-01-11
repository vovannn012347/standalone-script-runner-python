@echo OFF
echo --- Activating Virtual Environment ---
CALL .\venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Virtual environment not found or failed to activate.
    echo Please run 'python -m venv venv' first.
    goto :eof
)

echo --- Installing/Verifying Full Platform ---
pip install -r requirements.txt
pip install pyinstaller

echo --- Starting PyInstaller Build (This may take a long time...) ---

REM --noconfirm overwrites old builds
REM This one command reads your 'executor.spec' file and does all the work.
pyinstaller --noconfirm executor_dev.spec

if %ERRORLEVEL% neq 0 (
    echo.
    echo Build failed
    goto :finish
)

echo.
echo Build complete

set "SOURCE=src\executor.py"
set "TARGET=dist\executor\_internal\executor.py"

if exist "%SOURCE%" (
    if exist "dist\executor\_internal" (
        echo Копіюємо %SOURCE% → %TARGET%
        copy /Y "%SOURCE%" "%TARGET%" >nul
        if errorlevel 1 (
            echo Warning: could not copy executor.py
        ) else (
            echo executor.py copied to _internal
        )
    ) else (
        echo Warning: folder _internal not found, omitting copy
    )
) else (
    echo Warning: src\executor.py not found, omitting copy
)

REM Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat
echo --- Build script finished. ---