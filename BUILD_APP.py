#!/usr/bin/env python3
"""
🔨 TIFF Simulator V4.1 - Desktop App Builder
==============================================

Baut eine standalone Desktop-App mit PyInstaller.

Usage:
    python BUILD_APP.py

Output:
    dist/TIFF_Simulator_V4.1.exe  (oder .app auf macOS)
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

print("=" * 70)
print("🔨 TIFF SIMULATOR V4.1 - DESKTOP APP BUILD")
print("=" * 70)

# Check if PyInstaller is installed
try:
    import PyInstaller
    print(f"✓ PyInstaller version: {PyInstaller.__version__}")
except ImportError:
    print("\n❌ PyInstaller nicht installiert!")
    print("   Installiere mit: pip install pyinstaller")
    sys.exit(1)

# Check required files
required_files = [
    'tiff_simulator_gui_v4.py',
    'tiff_simulator_v3.py',
    'metadata_exporter.py',
    'batch_simulator.py',
    'track_analysis.py',
    'rf_trainer.py',
    'adaptive_rf_trainer.py',
    'diffusion_label_utils.py',
    'build_app.spec',
]

print("\n📋 Prüfe benötigte Dateien...")
missing = []
for f in required_files:
    if Path(f).exists():
        print(f"   ✓ {f}")
    else:
        print(f"   ❌ {f} - FEHLT!")
        missing.append(f)

if missing:
    print(f"\n❌ {len(missing)} Dateien fehlen!")
    print("   Build abgebrochen.")
    sys.exit(1)

print("\n✅ Alle Dateien vorhanden!")

# Clean old build
print("\n🧹 Cleanup alter Build-Dateien...")
for d in ['build', 'dist']:
    if Path(d).exists():
        shutil.rmtree(d)
        print(f"   Gelöscht: {d}/")

# Build with PyInstaller
print("\n🔨 Starte PyInstaller Build...")
print("   (Das kann 5-10 Minuten dauern...)")
print()

try:
    result = subprocess.run(
        ['pyinstaller', 'build_app.spec', '--clean', '--noconfirm'],
        check=True,
        capture_output=False
    )

    print("\n" + "=" * 70)
    print("✅ BUILD ERFOLGREICH!")
    print("=" * 70)

    # Find the executable
    if sys.platform == 'win32':
        exe_name = 'TIFF_Simulator_V4.1.exe'
    elif sys.platform == 'darwin':
        exe_name = 'TIFF_Simulator_V4.1.app'
    else:
        exe_name = 'TIFF_Simulator_V4.1'

    exe_path = Path('dist') / exe_name

    if exe_path.exists():
        print(f"\n📦 Executable erstellt:")
        print(f"   {exe_path.absolute()}")

        # Get size
        if exe_path.is_file():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"   Größe: {size_mb:.1f} MB")
        elif exe_path.is_dir():
            print(f"   (macOS .app Bundle)")

        print(f"\n🚀 Starte die App:")
        print(f"   cd dist")
        if sys.platform == 'win32':
            print(f"   {exe_name}")
        elif sys.platform == 'darwin':
            print(f"   open {exe_name}")
        else:
            print(f"   ./{exe_name}")

        print("\n💡 Tipp: Beim ersten Start mit console=True")
        print("   werden Fehler im Terminal angezeigt!")

    else:
        print(f"\n⚠️  Executable nicht gefunden: {exe_path}")
        print("   Build war erfolgreich, aber Datei fehlt?")

except subprocess.CalledProcessError as e:
    print("\n" + "=" * 70)
    print("❌ BUILD FEHLGESCHLAGEN!")
    print("=" * 70)
    print(f"\nFehler: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Prüfe ob alle Dependencies installiert sind:")
    print("      pip install -r requirements.txt")
    print("   2. Lösche build/ und dist/ manuell")
    print("   3. Versuche es nochmal")
    sys.exit(1)

print("\n" + "=" * 70)
print("🎉 DONE!")
print("=" * 70 + "\n")
