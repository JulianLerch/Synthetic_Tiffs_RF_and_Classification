@echo off
REM Build script for TIFF Simulator V4.0 Desktop App (Windows)
REM ===========================================================

echo.
echo 🔬 Building TIFF Simulator V4.0 Desktop Application
echo ====================================================
echo.

REM Check if PyInstaller is installed
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo ❌ PyInstaller not found!
    echo    Installing PyInstaller...
    pip install pyinstaller
    echo.
    echo ℹ️  PyInstaller installed. Continuing...
)

REM Clean previous builds
echo 🧹 Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build the application
echo.
echo 🔨 Building application with PyInstaller...
echo    (Dies kann einige Minuten dauern...)
echo.
python -m PyInstaller build_app.spec

REM Check if build was successful
if exist "dist\TIFF_Simulator_V4.exe" (
    echo.
    echo ✅ Build successful!
    echo.
    echo 📦 Application location: dist\TIFF_Simulator_V4.exe
    echo    File size: ~150-200 MB
    echo.
    echo 🚀 You can now run the application without Python installed!
    echo    Simply double-click: dist\TIFF_Simulator_V4.exe
    echo.
) else if exist "dist\TIFF_Simulator_V4\" (
    echo.
    echo ✅ Build successful!
    echo.
    echo 📦 Application location: dist\TIFF_Simulator_V4\
    echo.
    echo 🚀 To run: dist\TIFF_Simulator_V4\TIFF_Simulator_V4.exe
    echo.
) else (
    echo.
    echo ❌ Build failed! Check the output above for errors.
    echo.
    echo 💡 Troubleshooting:
    echo    1. Ensure all dependencies are installed: pip install -r requirements.txt
    echo    2. Try running manually: python -m PyInstaller build_app.spec
    echo    3. Check if tiff_simulator_gui_v4.py exists
    echo.
    pause
    exit /b 1
)

echo.
echo 📝 Note: The first run may take a few seconds to start.
echo.
pause
