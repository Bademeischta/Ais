@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   Ais Meta-Learning Arena - Lokaler Start
echo ============================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [FEHLER] Python wurde nicht gefunden.
    echo Bitte Python 3.10+ installieren: https://www.python.org/downloads/
    echo Bei der Installation "Add Python to PATH" aktivieren.
    pause
    exit /b 1
)

echo Prüfe Abhängigkeiten...
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo Erste Ausführung: Installiere Abhängigkeiten...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [FEHLER] pip install fehlgeschlagen.
        pause
        exit /b 1
    )
    echo.
)

echo Starte Server...
echo.
python app.py

pause
