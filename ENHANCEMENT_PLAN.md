# Weiterführende Enhancement-Ideen

## Physikalische Modellierung
- **Messdaten-gestützte z-Kalibrierung**: Die aktuelle z-Stack-Generierung plaziert Spots zufällig und verwendet konstante Intensitäten, wodurch ThunderSTORM-Referenzen schwer vergleichbar bleiben. Eine Grid-basierte Referenz mit definierter Intensitätsverteilung sowie importierbaren PSF-Profilen aus Bead-Messungen würde den Kalibrierungs-Workflow stabilisieren.【F:tiff_simulator_v3.py†L1596-L1689】
- **Deterministische Reproduzierbarkeit**: `generate_z_stack` und `generate_tiff` verlassen sich auf globale Zufallszahlen, was Debugging von Brechungsindex-Korrekturen erschwert. Ein optionaler Seed-Parameter oder ein dedizierter RNG-Kontext würde reproduzierbare Vergleichsreihen erlauben.【F:tiff_simulator_v3.py†L1614-L1673】
- **Physikalische Validierung der Brechungsindex-Korrektur**: Die heuristische `calculate_advanced_refractive_correction`-Formel kombiniert mehrere Faktoren ohne empirische Fits. Ein Vergleich mit Referenzdaten (z.B. PSFj oder Calibration Beads) und eine Möglichkeit, gemessene Skalierungs-Kurven zu laden, würden Unsicherheiten verringern.【F:tiff_simulator_v3.py†L169-L208】
- **SNR-Modelle für ThunderSTORM**: Aktuell wird die Intensitätsskala nur durch den PSF-Generator beeinflusst. Ein moduliertes SNR-Modell mit Kamera-spezifischen Transferfunktionen (EMCCD, sCMOS) könnte realistischere z-Stacks erzeugen.【F:tiff_simulator_v3.py†L1505-L1525】【F:tiff_simulator_v3.py†L1660-L1689】

## GUI-Verbesserungen
- **Architektur mit ViewModels**: Die GUI bündelt Business-Logik, Datenmodell und Widgets in einer einzigen Klasse. Eine Aufteilung in Services (Simulation, Export, Training) und ViewModels würde Wartbarkeit und Tests verbessern.【F:tiff_simulator_gui_v4.py†L35-L556】【F:tiff_simulator_gui_v4.py†L570-L835】
- **ThunderSTORM-orientierter Workflow**: Ein dediziertes Wizard-Layout (Schritt 1: Optik, Schritt 2: PSF Fit, Schritt 3: Export) samt Live-Restfehlern würde Anwender gezielt durch die Kalibrierung führen. Momentan sind die Einstellungen über mehrere Tabs verteilt.【F:tiff_simulator_gui_v4.py†L570-L835】
- **Barrierefreie Interaktion**: Große Button-Flächen, vereinfachte Tooltips und Tastaturkürzel könnten die Benutzung deutlich beschleunigen; derzeit dominieren Scroll-Formulare ohne Fokussteuerung.【F:tiff_simulator_gui_v4.py†L540-L835】

## Tooling & Validierung
- **Automatisierte Regressionstests**: Es fehlen Unit-Tests für die axialen Profile (`evaluate_z_profile`) und die Exportpfade. Ein kleines Set synthetischer Szenarien könnte sicherstellen, dass künftige Anpassungen physikalisch konsistent bleiben.【F:tiff_simulator_v3.py†L1727-L1741】
- **CLI/Batch-Integration für Kalibrierungen**: Ein Kommando wie `python batch_simulator.py --thunderstorm-calibration` könnte reproduzierbare Stacks + CSV-Kurven erzeugen und damit die GUI entlasten.【F:batch_simulator.py†L1-L40】
- **Dokumentations-Tutorials mit Vergleichsdaten**: Ergänzende Notebooks, die einen kompletten ThunderSTORM-Kalibrierlauf zeigen, würden Nutzern helfen, die neuen Modelle korrekt zu nutzen.【F:README.md†L380-L432】
