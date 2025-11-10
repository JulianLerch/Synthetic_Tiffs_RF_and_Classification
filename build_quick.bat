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
echo 🔨 Baue Application (One-File) inkl. neuer Z-Stack-Physik...
echo    Dies kann 2-5 Minuten dauern...
echo.

python -m PyInstaller ^
    --onefile ^
    --windowed ^
    --name=TIFF_Simulator_V4 ^
    --add-data="tiff_simulator_v3.py;." ^
    --add-data="metadata_exporter.py;." ^
    --add-data="batch_simulator.py;." ^
    --add-data="track_analysis.py;." ^
    --add-data="rf_trainer.py;." ^
    --add-data="rf_training_session.py;." ^
    --add-data="adaptive_rf_trainer.py;." ^
    --add-data="diffusion_label_utils.py;." ^
    --add-data="README.md;." ^
    --add-data="QUICKSTART.md;." ^
    --add-data="CHANGELOG.md;." ^
    --add-data="PHYSICS_VALIDATION.md;." ^
    --add-data="BATCH_MODE_GUIDE.md;." ^
    --add-data="RF_USAGE_GUIDE.md;." ^
    --add-data="TRACK_ANALYSIS_GUIDE.md;." ^
    --add-data="ADAPTIVE_RF_GUIDE.md;." ^
    --add-data="SETUP_GUIDE.md;." ^
    --hidden-import=PIL._tkinter_finder ^
    --hidden-import=tkinter ^
    --hidden-import=tkinter.filedialog ^
    --hidden-import=tkinter.messagebox ^
    --hidden-import=tkinter.ttk ^
    --hidden-import=numpy.core ^
    --hidden-import=scipy.stats ^
    --hidden-import=scipy.special ^
    --hidden-import=sklearn ^
    --hidden-import=sklearn.ensemble ^
    --hidden-import=sklearn.tree ^
    --hidden-import=joblib ^
    --hidden-import=matplotlib ^
    --hidden-import=matplotlib.backends.backend_tkagg ^
    --hidden-import=matplotlib.backends.tkagg ^
    --hidden-import=matplotlib.backends._tkagg ^
    --hidden-import=matplotlib.backends.backend_agg ^
    --hidden-import=matplotlib.figure ^
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
    echo 🆕 Enthaltene Features:
    echo    • ThunderSTORM Z-Stack Preset mit realistischer Brechungsindex-Korrektur
    echo    • Live Z-Profil Vorschau (Matplotlib) fuer Feinjustierung
    echo    • Erweiterte Guides (README, QUICKSTART, PHYSICS_VALIDATION, ...)
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
