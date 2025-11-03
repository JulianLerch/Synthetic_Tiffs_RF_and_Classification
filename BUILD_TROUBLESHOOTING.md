# 🛠️ BUILD TROUBLESHOOTING - TIFF Simulator V4.0

## Problem: "pyinstaller command not found" (Windows)

### ✅ LÖSUNG: Korrigiertes Build-Script verwenden

Ich habe das Build-Script korrigiert. Probieren Sie erneut:

```cmd
build_desktop_app.bat
```

Das Script verwendet jetzt `python -m PyInstaller` statt nur `pyinstaller`.

---

## Alternative: Quick Build (Einfacher!)

Falls das normale Build-Script Probleme macht:

```cmd
build_quick.bat
```

**Vorteile:**
- ✅ Einfacher (keine .spec Datei)
- ✅ One-File Mode (eine einzelne .exe)
- ✅ Funktioniert fast immer

**Nachteile:**
- ⚠️ Langsamer Start (3-5 Sekunden)
- ⚠️ Größer (~200 MB statt ~150 MB)

---

## Manuelle Build-Methode

Falls beide Scripts Probleme machen:

```cmd
# 1. Dependencies installieren
pip install -r requirements.txt

# 2. PyInstaller installieren
pip install pyinstaller

# 3. Manuell bauen (Simple)
python -m PyInstaller --onefile --windowed --name=TIFF_Simulator_V4 tiff_simulator_gui_v4.py

# ODER: Mit allen Dateien
python -m PyInstaller --onefile --windowed --name=TIFF_Simulator_V4 ^
    --add-data="tiff_simulator_v3.py;." ^
    --add-data="metadata_exporter.py;." ^
    --add-data="batch_simulator.py;." ^
    tiff_simulator_gui_v4.py
```

Die fertige App ist dann in: `dist\TIFF_Simulator_V4.exe`

---

## Häufige Fehler & Lösungen

### 1. "ModuleNotFoundError: No module named 'numpy'"

**Problem:** Dependencies nicht installiert

**Lösung:**
```cmd
pip install -r requirements.txt
```

### 2. "FileNotFoundError: tiff_simulator_gui_v4.py"

**Problem:** Sie sind im falschen Ordner

**Lösung:**
```cmd
cd C:\Pfad\zu\MasterthesisCNN
dir  # Prüfen ob Dateien da sind
build_desktop_app.bat
```

### 3. Build dauert sehr lange / friert ein

**Normal!** PyInstaller-Builds können 2-5 Minuten dauern.

**Anzeichen dass es funktioniert:**
- Viele Zeilen mit "INFO: ..."
- CPU-Auslastung hoch
- Ordner `build/` und `dist/` werden erstellt

**Geduld haben!** ⏳

### 4. "ImportError: DLL load failed" beim Starten der .exe

**Problem:** Antivirus blockiert oder fehlende Windows-Updates

**Lösung:**
- Windows Update ausführen
- .exe zur Antivirus-Whitelist hinzufügen
- Visual C++ Redistributable installieren: https://aka.ms/vs/17/release/vc_redist.x64.exe

### 5. PyInstaller generiert Ordner statt .exe

**Normal!** Zwei Modi möglich:

**One-File Mode:** Eine .exe
```
dist/
  └── TIFF_Simulator_V4.exe  (150-200 MB)
```

**One-Folder Mode:** Ordner mit .exe
```
dist/
  └── TIFF_Simulator_V4/
      ├── TIFF_Simulator_V4.exe
      ├── python313.dll
      └── ... (viele Dateien)
```

Beide funktionieren! One-File ist portabler, One-Folder startet schneller.

---

## Schnellste Alternative: Python direkt nutzen

Wenn Build-Probleme zu nervig sind, **nutzen Sie einfach Python direkt:**

```cmd
# Einmalig: Dependencies installieren
pip install -r requirements.txt

# Jedes Mal: Starten
python START_SIMULATOR.py
```

**Vorteile:**
- ✅ Kein Build nötig
- ✅ Schnellerer Start
- ✅ Einfacher zu debuggen

**Nachteil:**
- ⚠️ Python muss installiert sein

---

## System-Anforderungen

**Minimum:**
- Windows 10/11 (64-bit)
- Python 3.8+ (falls kein Executable)
- 4 GB RAM
- 500 MB freier Speicher

**Empfohlen:**
- Windows 11 (64-bit)
- Python 3.10 oder 3.11
- 8 GB RAM
- 1 GB freier Speicher

---

## Erfolg prüfen

Nach dem Build:

```cmd
# Prüfe ob .exe existiert
dir dist

# Teste die .exe
cd dist
TIFF_Simulator_V4.exe
```

**Erwartetes Verhalten:**
1. Kurzes Laden (2-5 Sekunden beim ersten Mal)
2. GUI öffnet sich
3. Header zeigt "V4.0"

---

## Support-Checklist

Wenn nichts funktioniert, prüfen Sie:

- ✅ Python installiert? `python --version`
- ✅ Im richtigen Ordner? `dir` zeigt die .py Dateien?
- ✅ Dependencies installiert? `pip list | findstr numpy`
- ✅ Genug Speicherplatz? (mind. 500 MB)
- ✅ Antivirus aus? (temporär zum Testen)
- ✅ Windows aktuell? (Windows Update)

---

## Alternative: Direkt in Python arbeiten

**Für Ihre Masterthesis ist es völlig OK, Python direkt zu nutzen!**

```cmd
# Setup (einmalig):
git clone https://github.com/JulianLerch/MasterthesisCNN.git
cd MasterthesisCNN
git checkout claude/project-analysis-documentation-011CUbZyiQGDBZrxSBr7AD14
pip install -r requirements.txt

# Nutzen (jedes Mal):
python START_SIMULATOR.py
```

Das ist sogar schneller und flexibler als ein Executable!

---

## Zusammenfassung

**3 Wege zum Ziel:**

1. **Build-Script** (empfohlen wenn's klappt):
   ```cmd
   build_desktop_app.bat
   ```

2. **Quick Build** (einfache Alternative):
   ```cmd
   build_quick.bat
   ```

3. **Python direkt** (am einfachsten!):
   ```cmd
   python START_SIMULATOR.py
   ```

**Für Ihre Thesis reicht Option 3 völlig aus!** 🎓

Die Desktop-App ist nur ein "Nice-to-Have" wenn Sie die Software weitergeben wollen.
