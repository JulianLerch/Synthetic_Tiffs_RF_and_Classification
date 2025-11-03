@echo off
REM Quick Build Script - Einfache Alternative ohne .spec Datei
REM ===========================================================

echo.
echo 🔬 TIFF Simulator V4.0 - Quick Build (One-File Mode)
echo ====================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python nicht gefunden!
    echo    Bitte installieren Sie Python von python.org
    pause
    exit /b 1
)

echo ✅ Python gefunden
echo.

REM Install PyInstaller if needed
echo 📦 Prüfe PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo    Installing PyInstaller...
    pip install pyinstaller
)
echo ✅ PyInstaller bereit
echo.

REM Clean
echo 🧹 Räume auf...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo.

REM Build
echo 🔨 Baue Application (One-File)...
echo    Dies kann 2-5 Minuten dauern...
echo.

python -m PyInstaller ^
    --onefile ^
    --windowed ^
    --name=TIFF_Simulator_V4 ^
    --add-data="tiff_simulator_v3.py;." ^
    --add-data="metadata_exporter.py;." ^
    --add-data="batch_simulator.py;." ^
    --hidden-import=PIL._tkinter_finder ^
    --hidden-import=numpy.core ^
    tiff_simulator_gui_v4.py

echo.

REM Check result
if exist "dist\TIFF_Simulator_V4.exe" (
    echo.
    echo ========================================
    echo ✅ BUILD ERFOLGREICH!
    echo ========================================
    echo.
    echo 📦 Anwendung: dist\TIFF_Simulator_V4.exe
    echo 💾 Größe: ~150-200 MB (One-File)
    echo.
    echo 🚀 FERTIG! Einfach doppelklicken:
    echo    dist\TIFF_Simulator_V4.exe
    echo.
    echo 📝 Hinweis: Erster Start kann 3-5 Sekunden dauern
    echo            (entpackt temporäre Dateien)
    echo.
) else (
    echo ❌ Build fehlgeschlagen!
    echo.
    echo Mögliche Probleme:
    echo 1. Fehlende Dateien? Prüfe ob vorhanden:
    echo    - tiff_simulator_gui_v4.py
    echo    - tiff_simulator_v3.py
    echo    - metadata_exporter.py
    echo    - batch_simulator.py
    echo.
    echo 2. Dependencies? Installiere:
    echo    pip install -r requirements.txt
    echo.
)

pause
