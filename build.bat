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
pyinstaller --noconfirm executor.spec

IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyInstaller build failed.
) ELSE (
    echo --- Build complete! Standalone platform is in 'dist\executor' folder. ---
)

REM Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat
echo --- Build script finished. ---