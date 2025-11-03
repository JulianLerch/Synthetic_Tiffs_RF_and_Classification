#!/usr/bin/env python3
"""
🚀 QUICK START - TIFF SIMULATOR V3.0
====================================

Dieses Skript startet die GUI automatisch und prüft alle Dependencies.

Verwendung:
    python START.py

Oder doppelklick auf START.py (Windows/macOS)
"""

import sys
import os

# ASCII Art Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   🔬 HYPERREALISTISCHER TIFF-SIMULATOR V3.0                      ║
║                                                                   ║
║   Wissenschaftlich präzise Simulation von                         ║
║   Single-Molecule Tracking Daten                                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""

def check_python_version():
    """Prüft Python-Version."""
    if sys.version_info < (3, 8):
        print("❌ FEHLER: Python 3.8 oder höher erforderlich!")
        print(f"   Aktuelle Version: {sys.version}")
        print("\n   Bitte aktualisiere Python:")
        print("   https://www.python.org/downloads/")
        return False
    
    print(f"✅ Python Version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Prüft ob alle Dependencies installiert sind."""
    
    required_packages = {
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'tkinter': 'tkinter (GUI)'
    }
    
    missing = []
    
    for package, name in required_packages.items():
        try:
            if package == 'PIL':
                import PIL
            elif package == 'tkinter':
                import tkinter
            else:
                __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NICHT INSTALLIERT")
            missing.append(name)
    
    if missing:
        print(f"\n❌ FEHLER: {len(missing)} Pakete fehlen!")
        print("\n📦 Installation:")
        print("   pip install -r requirements.txt")
        
        if 'tkinter (GUI)' in missing:
            print("\n   tkinter Installation:")
            print("   Ubuntu/Debian: sudo apt-get install python3-tk")
            print("   Fedora/RHEL: sudo dnf install python3-tkinter")
            print("   macOS: Built-in (sollte bereits installiert sein)")
            print("   Windows: Built-in (sollte bereits installiert sein)")
        
        return False
    
    return True

def check_files():
    """Prüft ob alle benötigten Dateien vorhanden sind."""
    
    required_files = [
        'tiff_simulator_v3.py',
        'metadata_exporter.py',
        'batch_simulator.py',
        'tiff_simulator_gui.py'
    ]
    
    missing = []
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} - NICHT GEFUNDEN")
            missing.append(filename)
    
    if missing:
        print(f"\n❌ FEHLER: {len(missing)} Dateien fehlen!")
        print("\n   Bitte stelle sicher, dass alle Dateien im gleichen Ordner sind:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    return True

def start_gui():
    """Startet die GUI."""
    
    print("\n" + "="*70)
    print("🚀 STARTE GUI...")
    print("="*70 + "\n")
    
    try:
        # Importiere und starte GUI
        from tiff_simulator_gui import TIFFSimulatorGUI
        import tkinter as tk
        
        root = tk.Tk()
        app = TIFFSimulatorGUI(root)
        root.mainloop()
        
    except Exception as e:
        print(f"\n❌ FEHLER beim Starten der GUI:")
        print(f"   {str(e)}")
        print("\n   Mögliche Lösungen:")
        print("   1. Prüfe ob alle Dependencies installiert sind")
        print("   2. Prüfe ob tkinter funktioniert: python -m tkinter")
        print("   3. Bei Linux: Installiere python3-tk")
        return False
    
    return True

def main():
    """Hauptfunktion."""
    
    print(BANNER)
    print("\n🔍 SYSTEM-CHECK")
    print("="*70)
    
    # 1. Python Version prüfen
    if not check_python_version():
        input("\nDrücke Enter zum Beenden...")
        return
    
    print("\n📦 DEPENDENCIES")
    print("="*70)
    
    # 2. Dependencies prüfen
    if not check_dependencies():
        input("\nDrücke Enter zum Beenden...")
        return
    
    print("\n📄 DATEIEN")
    print("="*70)
    
    # 3. Dateien prüfen
    if not check_files():
        input("\nDrücke Enter zum Beenden...")
        return
    
    # 4. GUI starten
    start_gui()
    
    print("\n✅ Programm beendet. Bis bald! 👋")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Abgebrochen durch Benutzer (Ctrl+C)")
        print("Bis bald! 👋")
    except Exception as e:
        print(f"\n❌ UNERWARTETER FEHLER:")
        print(f"   {str(e)}")
        print("\n   Bitte öffne ein Issue oder kontaktiere den Support.")
        input("\nDrücke Enter zum Beenden...")
