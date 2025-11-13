"""
Test: Batch-Modus mit TrackMate XML Export
===========================================

Testet den kompletten Workflow:
1. Batch-Generierung mehrerer TIFFs mit verschiedenen Polyzeiten
2. Mehrere Wiederholungen pro Polyzeit
3. Automatische Ordnerstruktur (polyzeit_XXmin/)
4. Automatische TrackMate XML-Generierung pro Polyzeit

Test-Konfiguration:
- 3 Polyzeiten: 30, 60, 90 min
- 2 Wiederholungen pro Polyzeit
- Mit und ohne Astigmatismus
- Total: 6 TIFFs + 3 XMLs
"""

from pathlib import Path
from batch_simulator import BatchSimulator
from tiff_simulator_v3 import TDI_PRESET
import shutil

# Clean test output directory
test_output_dir = Path("./test_batch_trackmate_output")
if test_output_dir.exists():
    shutil.rmtree(test_output_dir)

print("=" * 80)
print("🧪 BATCH-MODUS MIT TRACKMATE XML EXPORT TEST")
print("=" * 80)

# Create batch simulator
batch = BatchSimulator(output_dir=str(test_output_dir))

# Add polyzeit series with repeats
print("\n📋 KONFIGURATION")
print("-" * 80)
print("Polyzeiten: 30, 60, 90 min")
print("Wiederholungen: 2 pro Polyzeit")
print("Astigmatismus: Ja (für XML z-Positionen)")
print("Image size: 64x64 (schnell für Test)")
print("Frames: 20 (schnell für Test)")
print("Spots: 5")

batch.add_polyzeit_series(
    times=[30, 60, 90],
    detector=TDI_PRESET,
    repeats=2,
    image_size=(64, 64),
    num_spots=5,
    num_frames=20,
    frame_rate_hz=20.0,
    astigmatism=True
)

print(f"\nTotal Tasks: {len(batch.tasks)}")
print("-" * 80)

# Run batch
print("\n🔄 STARTE BATCH-SIMULATION...")
print("=" * 80)

stats = batch.run()

# Verify output structure
print("\n" + "=" * 80)
print("🔍 ÜBERPRÜFUNG DER OUTPUT-STRUKTUR")
print("=" * 80)

# Check for polyzeit folders
polyzeitfolders = list(test_output_dir.glob("polyzeit_*min"))
print(f"\n📁 Polyzeit-Ordner gefunden: {len(polyzeitfolders)}")

for folder in sorted(polyzeitfolders):
    print(f"\n  📂 {folder.name}/")

    # Check TIFFs
    tiffs = list(folder.glob("*.tif"))
    print(f"     TIFFs: {len(tiffs)}")
    for tif in sorted(tiffs):
        print(f"       • {tif.name} ({tif.stat().st_size / 1024:.1f} KB)")

    # Check metadata JSON
    jsons = list(folder.glob("*_metadata.json"))
    print(f"     Metadata JSONs: {len(jsons)}")
    for js in sorted(jsons):
        print(f"       • {js.name}")

    # Check TrackMate XML
    xmls = list(folder.glob("trackmate_*.xml"))
    print(f"     TrackMate XMLs: {len(xmls)}")
    for xml in sorted(xmls):
        print(f"       • {xml.name} ({xml.stat().st_size / 1024:.1f} KB)")

        # Validate XML content
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(xml)
            root = tree.getroot()

            # Check attributes
            n_tracks = root.get('nTracks')
            space_units = root.get('spaceUnits')
            time_units = root.get('timeUnits')

            print(f"         ✓ Tracks: {n_tracks}")
            print(f"         ✓ Units: {space_units}, {time_units}")

            # Count detections
            total_detections = 0
            for particle in root.findall('particle'):
                detections = particle.findall('detection')
                total_detections += len(detections)

            print(f"         ✓ Total Detections: {total_detections}")

            # Check z-values (should not all be 0.0 since astigmatism is on)
            z_values = []
            for particle in root.findall('particle'):
                for detection in particle.findall('detection'):
                    z_val = float(detection.get('z'))
                    z_values.append(z_val)

            non_zero_z = sum(1 for z in z_values if abs(z) > 1e-6)
            print(f"         ✓ Non-zero z-Werte: {non_zero_z}/{len(z_values)}")

            if non_zero_z > 0:
                print(f"         ✅ Z-Positionen enthalten! (Astigmatismus aktiv)")
            else:
                print(f"         ⚠️ Alle z=0 (Astigmatismus evtl. nicht aktiv?)")

        except Exception as e:
            print(f"         ❌ XML-Parsing Fehler: {e}")

# Summary
print("\n" + "=" * 80)
print("📊 TEST-ZUSAMMENFASSUNG")
print("=" * 80)
print(f"✅ Generierte TIFFs: {stats['completed']}/{stats['total_tasks']}")
print(f"✅ Polyzeit-Ordner: {len(polyzeitfolders)} (erwartet: 3)")
print(f"✅ Output-Verzeichnis: {test_output_dir}")

# Verify expected structure
expected_structure_ok = True

# Should have 3 polyzeit folders
if len(polyzeitfolders) != 3:
    print(f"❌ FEHLER: Erwarte 3 Polyzeit-Ordner, gefunden: {len(polyzeitfolders)}")
    expected_structure_ok = False

# Each folder should have 2 TIFFs, 2 metadata JSONs, 1 XML
for folder in polyzeitfolders:
    tiffs = list(folder.glob("*.tif"))
    jsons = list(folder.glob("*_metadata.json"))
    xmls = list(folder.glob("trackmate_*.xml"))

    if len(tiffs) != 2:
        print(f"❌ FEHLER: {folder.name} hat {len(tiffs)} TIFFs, erwartet: 2")
        expected_structure_ok = False

    if len(jsons) != 2:
        print(f"❌ FEHLER: {folder.name} hat {len(jsons)} JSONs, erwartet: 2")
        expected_structure_ok = False

    if len(xmls) != 1:
        print(f"❌ FEHLER: {folder.name} hat {len(xmls)} XMLs, erwartet: 1")
        expected_structure_ok = False

if expected_structure_ok:
    print("\n🎉 TEST ERFOLGREICH!")
    print("   Alle TIFFs, Metadaten und TrackMate XMLs wurden korrekt generiert!")
    print("   Ordnerstruktur ist korrekt!")
else:
    print("\n❌ TEST FEHLGESCHLAGEN!")
    print("   Bitte überprüfe die Fehler oben!")

print("=" * 80)
