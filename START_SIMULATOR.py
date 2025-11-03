#!/usr/bin/env python3
"""
🚀 TIFF SIMULATOR V4.0 - LAUNCHER
==================================

Einfacher Launcher mit Dependency-Check und automatischer Installation.

Features:
- Prüft alle Dependencies
- Installiert fehlende Pakete automatisch
- Startet die optimierte GUI V4.0
- Benutzerfreundliche Fehlermeldungen

Version: 4.0
"""

import sys
import subprocess
import importlib.util

# Required packages
REQUIRED_PACKAGES = {
    'numpy': 'numpy>=1.21.0',
    'PIL': 'Pillow>=9.2.0',
    'tqdm': 'tqdm>=4.64.0',
}

# Optional packages
OPTIONAL_PACKAGES = {
    'matplotlib': 'matplotlib>=3.5.0',
}


def check_package(package_name):
    """Prüft ob ein Paket installiert ist."""
    return importlib.util.find_spec(package_name) is not None


def install_package(package_spec):
    """Installiert ein Paket via pip."""
    print(f"   📦 Installiere {package_spec}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Hauptfunktion."""
    print("🔬 TIFF Simulator V4.0 - Starting...")
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

    # Check & install required packages
    print("📋 Prüfe Dependencies...")
    missing_packages = []

    for package, spec in REQUIRED_PACKAGES.items():
        if not check_package(package):
            missing_packages.append(spec)
            print(f"   ⚠️  {package} nicht gefunden")
        else:
            print(f"   ✅ {package}")

    # Install missing packages
    if missing_packages:
        print()
        print("🔧 Installiere fehlende Pakete...")
        for spec in missing_packages:
            if not install_package(spec):
                print(f"\n❌ Installation von {spec} fehlgeschlagen!")
                print("   Bitte installiere manuell mit:")
                print(f"   pip install {spec}")
                input("\nDrücke Enter zum Beenden...")
                sys.exit(1)
        print("✅ Alle Pakete installiert!")

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
    print("🚀 Starte TIFF Simulator V4.0 GUI...")
    print("=" * 50)
    print()

    # Start GUI
    try:
        # Try V4.0 first, fallback to V3.0
        try:
            from tiff_simulator_gui_v4 import TIFFSimulatorGUI_V4
            import tkinter as tk
            root = tk.Tk()
            app = TIFFSimulatorGUI_V4(root)
            print("✅ GUI V4.0 (Advanced Edition) geladen!")
            root.mainloop()
        except ImportError:
            print("⚠️  GUI V4.0 nicht gefunden, verwende V3.0...")
            from tiff_simulator_gui import TIFFSimulatorGUI
            import tkinter as tk
            root = tk.Tk()
            app = TIFFSimulatorGUI(root)
            root.mainloop()

    except ImportError as e:
        print(f"\n❌ Import-Fehler: {e}")
        print("\nBitte stelle sicher, dass alle Dateien vorhanden sind:")
        print("   - tiff_simulator_v3.py")
        print("   - tiff_simulator_gui_v4.py (oder tiff_simulator_gui.py)")
        print("   - metadata_exporter.py")
        print("   - batch_simulator.py")
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
