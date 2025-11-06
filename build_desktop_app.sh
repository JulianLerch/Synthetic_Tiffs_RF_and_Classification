#!/bin/bash
# Build script for TIFF Simulator V4.0 Desktop App
# =================================================

echo "🔬 Building TIFF Simulator V4.0 Desktop Application"
echo "===================================================="
echo ""

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "❌ PyInstaller not found!"
    echo "   Installing PyInstaller..."
    pip install pyinstaller
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/

# Build the application
echo ""
echo "🔨 Building application with PyInstaller..."
pyinstaller build_app.spec

# Check if build was successful (handles V4 and V4.1 artifact names)
if [ -f "dist/TIFF_Simulator_V4.1.exe" ] || [ -d "dist/TIFF_Simulator_V4.1" ] \
   || [ -f "dist/TIFF_Simulator_V4.exe" ] || [ -d "dist/TIFF_Simulator_V4" ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "📦 Application location:"
    if [ -f "dist/TIFF_Simulator_V4.1.exe" ]; then
        echo "   dist/TIFF_Simulator_V4.1.exe"
    elif [ -d "dist/TIFF_Simulator_V4.1" ]; then
        echo "   dist/TIFF_Simulator_V4.1/"
    elif [ -d "dist/TIFF_Simulator_V4" ]; then
        echo "   dist/TIFF_Simulator_V4/"
    else
        echo "   dist/TIFF_Simulator_V4.exe"
    fi
    echo ""
    echo "🚀 You can now run the application without Python installed!"
else
    echo ""
    echo "❌ Build failed! Check the output above for errors."
    exit 1
fi
