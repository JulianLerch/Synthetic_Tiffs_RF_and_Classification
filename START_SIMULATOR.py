#!/usr/bin/env python3
"""
🚀 TIFF SIMULATOR V5.0 - LAUNCHER
==================================

Einfacher Launcher mit Dependency-Check für die moderne GUI

Features:
- Prüft alle Dependencies
- Benutzerfreundliche Fehlermeldungen
- Startet die moderne GUI V5.0

Version: 5.0 - November 2025
"""

import sys
import importlib.util

# Required packages
REQUIRED_PACKAGES = {
    'numpy': 'numpy>=1.21.0',
    'PIL': 'Pillow>=9.2.0',
    'scipy': 'scipy>=1.8.0',
    'tqdm': 'tqdm>=4.64.0',
}

# Optional packages
OPTIONAL_PACKAGES = {
    'matplotlib': 'matplotlib>=3.5.0',
}


def check_package(package_name):
    """Prüft ob ein Paket installiert ist."""
    return importlib.util.find_spec(package_name) is not None


def main():
    """Hauptfunktion."""
    print("🔬 TIFF Simulator V5.0 - Starting...")
    print("=" * 50)
    print()

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 oder höher erforderlich!")
        print(f"   Aktuelle Version: {sys.version}")
        input("\nDrücke Enter zum Beenden...")
        sys.exit(1)

    print("✅ Python Version:", sys.version.split()[0])
    print()

    # Check required packages
    print("📋 Prüfe Dependencies...")
    missing_packages = []

    for package, spec in REQUIRED_PACKAGES.items():
        if not check_package(package):
            missing_packages.append(spec)
            print(f"   ⚠️  {package} nicht gefunden")
        else:
            print(f"   ✅ {package}")

    # Check tkinter
    try:
        import tkinter
        print(f"   ✅ tkinter")
    except ImportError:
        print(f"   ❌ tkinter nicht gefunden!")
        print("\n   Installation:")
        print("   Ubuntu/Debian: sudo apt-get install python3-tk")
        print("   Fedora/RHEL: sudo dnf install python3-tkinter")
        input("\nDrücke Enter zum Beenden...")
        sys.exit(1)

    # Check if packages are missing
    if missing_packages:
        print()
        print("❌ Fehlende Pakete!")
        print("\nBitte installiere die fehlenden Pakete mit:")
        print("  pip install -r requirements.txt")
        print("\nOder einzeln:")
        for spec in missing_packages:
            print(f"  pip install {spec}")
        input("\nDrücke Enter zum Beenden...")
        sys.exit(1)

    # Check optional packages
    print()
    print("📋 Optionale Pakete:")
    for package, spec in OPTIONAL_PACKAGES.items():
        if check_package(package):
            print(f"   ✅ {package}")
        else:
            print(f"   ⚠️  {package} nicht installiert (optional)")

    print()
    print("=" * 50)
    print("🚀 Starte TIFF Simulator V5.0 GUI...")
    print("=" * 50)
    print()

    # Start GUI
    try:
        from tiff_simulator_gui import main as gui_main
        gui_main()

    except ImportError as e:
        print(f"\n❌ Import-Fehler: {e}")
        print("\nBitte stelle sicher, dass alle Dateien vorhanden sind:")
        print("   - tiff_simulator_v3.py")
        print("   - tiff_simulator_gui.py")
        print("   - batch_simulator.py")
        print("   - metadata_exporter.py")
        input("\nDrücke Enter zum Beenden...")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Fehler beim Starten: {e}")
        import traceback
        traceback.print_exc()
        input("\nDrücke Enter zum Beenden...")
        sys.exit(1)


if __name__ == "__main__":
    main()
