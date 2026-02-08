@echo off
REM ============================================================
REM DVC Setup Script for Windows
REM Cats & Dogs Classification MLOps Project
REM ============================================================

REM Change to project root directory (parent of scripts folder)
cd /d "%~dp0.."

echo ============================================================
echo DVC SETUP SCRIPT
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

echo [1/5] Checking DVC installation...
dvc version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo DVC is not installed. Installing DVC...
    pip install dvc
) ELSE (
    echo DVC is already installed.
)

echo.
echo [2/5] Initializing DVC repository...
python src\dvc_scripts\init_dvc.py
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: DVC initialization failed
    pause
    exit /b 1
)

echo.
echo [3/5] Verifying DVC setup...
dvc status
IF %ERRORLEVEL% NEQ 0 (
    echo WARNING: DVC status check failed
)

echo.
echo [4/5] Displaying pipeline DAG...
dvc dag
IF %ERRORLEVEL% NEQ 0 (
    echo WARNING: Could not display pipeline DAG
)

echo.
echo [5/5] Setup complete!
echo.
echo ============================================================
echo DVC SETUP COMPLETED SUCCESSFULLY
echo ============================================================
echo.
echo Next steps:
echo   1. Run data ingestion: dvc repro data_ingestion
echo   2. Run full pipeline: dvc repro
echo   3. Or use: python src\dvc_scripts\run_pipeline.py --help
echo.

pause
