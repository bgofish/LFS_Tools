@echo off
setlocal enabledelayedexpansion

:: --- CONFIGURATION ---
:: Ensure this matches your actual python filename
set "PY_SCRIPT=PyVista_ColmapBIN-CRS.py"
:: ---------------------

title COLMAP Explorer Launcher

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in your PATH.
    pause
    exit /b
)

echo [2/3] Verifying required packages...
:: List of packages needed
set "PACKAGES=pyvista pyvistaqt PyQt5 numpy"

for %%P in (%PACKAGES%) do (
    python -c "import %%P" >nul 2>&1
    if !errorlevel! neq 0 (
        echo Installing %%P...
        pip install %%P --quiet
    ) else (
        echo %%P is already installed.
    )
)

echo [3/3] Launching %PY_SCRIPT%...
echo.
python "%PY_SCRIPT%"

if %errorlevel% neq 0 (
    echo.
    echo Application closed with an error (Code: %errorlevel%).
    pause
)
