1) Recherche & Evidenz (Physik/Chemie/Mathematik)

Grundsatz: Vor jedem fachlichen Schritt ist eine Literaturrecherche durchzuführen. Es gelten ausschließlich universitär verwendbare Quellen:

Zulässig: Peer-reviewte Paper (Verlagsseiten, arXiv nur mit zusätzlicher Verlagsreferenz), Konferenzbeiträge, Lehrbücher (Universitätsverlage), Universitätsskripte/Handouts.

Nicht zulässig: Wikipedia, Blogs, „graue“ Websites, Foren, Marketing-Whitepaper, generische Q&A-Seiten.

Muss-Anforderungen:

Jede fachliche Aussage (Formel, Konstanten, Annahme, Verfahren) bekommt mindestens eine Primärquelle.

Quellenangaben in docs/CITATIONS.bib (BibTeX) und Kurzreferenzen im passenden Modul/Notebook.

Bei konkurrierenden Modellen: Unterschiede, Gültigkeitsbereiche, Annahmen dokumentieren.

Einheitliche Einheiten (SI), Dimensionsanalyse für zentrale Gleichungen.

Numerische Validierung: Referenzbeispiele aus der Literatur nachrechnen und als Tests beilegen.

Minimaler Recherche-Output (pro Feature/Fix):

/docs/research/<feature-id>/
  summary.md          # 5–10 Sätze: Fragestellung, Kernergebnis, Limitationen
  sources.bib         # BibTeX-Quellen
  notes.md            # wichtige Ableitungen/Annahmen
  validations.ipynb   # Nachrechnungen, Vergleich zu Paper-Grafiken/Tables

2) Arbeitsablauf (verbindlich)

Planen: Ticket lesen → Hypothesen/Annahmen notieren → Messgrößen & Akzeptanzkriterien definieren.

Recherchieren: Quellen sammeln (s.o.) → summary.md erstellen.

Entwurf: API-Skizze, Datenfluss, GUI-Flows, Fehlerszenarien, Performanceziele.

Implementieren: Vollständiger Code inkl. Tests, Typisierung, Linting, Logging.

Validieren: Unit-/Property-/Golden-Tests, Benchmarks, GUI-Usability-Checks, fachliche Validierung vs. Literatur.

Dokumentieren: README.md aktualisieren (siehe Abschnitt 4), Modul-Docs, Changelog.

Integration: Build-Pipelines anpassen, Artefakte erzeugen, Beispiele/demos aktualisieren.

Review intern (Agent-Self-Review): Checkliste (Abschnitt 9) abarbeiten, erst dann PR.

3) Coding-Standards

Sprachen/Frameworks: Gemäß Projektvorgaben (Python/TypeScript etc.). Strikte Typisierung (Python: mypy, TS: strict).

Linting/Format: ruff/black (Python), eslint/prettier (TS). Build bricht bei Verstößen.

Tests: pytest/vitest o.ä., ≥90 % relevante Pfadabdeckung; Property-Based Tests für numerische Kerne.

Logging & Fehlerbehandlung: strukturierte Logs, klare Fehlermeldungen mit Lösungshinweisen.

Reproduzierbarkeit: deterministische Seeds, feste Versionsranges (uv/poetry bzw. pnpm), .lock-Dateien committen.

Performance: Komplexität begründen, Benchmarks ablegen (Abschnitt 8).

Sicherheit/IP: Lizenzen prüfen, keine geschützten Datensätze ohne Freigabe, Secrets nur via .env/Vault.

4) README.md – immer aktuell halten (Pflicht)

Bei jedem Commit prüfen/aktualisieren:

Projektüberblick (1 Abs.), Features, Systemvoraussetzungen.

Installation & Build (Copy-Paste-fähig):

Backend/Frontend Setup

Daten/Modelle laden/erzeugen

Build/Release-Befehle

Schnellstart mit minimalem Beispiel (CLI & GUI).

Konfiguration (.env.example, Flags).

Validierung: Wie wissenschaftliche Tests/Notebooks ausgeführt werden.

Zitation: Verweis auf docs/CITATIONS.bib.

Changelog (Kurzverweis auf CHANGELOG.md).

Automatik (empfohlen):

scripts/sync_readme.py generiert Teile (CLI-Hilfe, Env-Variablen, Modul-Docs) → README einfügen.

5) GUI/UX – Benutzerfreundlichkeit (verbindlich)

Konsistentes Layout, klare Hierarchie, selbsterklärende Labels/Tooltips.

Fehlervermeidung: Validierung bei Eingabe, sinnvolle Defaults, Undo/Reset.

Zugänglichkeit (a11y): Tastaturbedienung, Screen-Reader-Labels, ausreichende Kontraste.

Feedback: Ladeindikator, Fortschrittsbalken, Erfolg/Fehler-Toasts mit Handlungstipps.

Leistung: Keine UI-Blocks; lange Jobs asynchron mit Abbruchmöglichkeit.

Internationalisierung: mind. Deutsch/Englisch vorbereitet.

Persistenz: Nutzer-Presets/Zuletzt benutzt speichern.

Dokumentierte Workflows: Onboarding-Tour oder „Erste Schritte“.

GUI-Abnahme-kriterien:

Ein unerfahrener Nutzer kann den Schnellstart-Use-Case in < 3 Klicks ausführen.

Keine unbeschrifteten Icons; jede Aktion ist discoverable.

Mindest-Kontrast WCAG AA, Tab-Reihenfolge logisch.

6) App-Integration & Build

Struktur (Beispiel):

/app
  /backend
  /frontend
  /gui
  /docs
  /scripts
  /tests
  /benchmarks


Build/CI (Muss):

CI-Pipeline (z.B. GitHub Actions):

Lint & Typecheck

Unit/Property/Integration-Tests

Build Artefakte (Wheel/Docker/Static)

Beispiel-Pipelines/Notebooks ausführen

README-Sync & Artefakt-Upload (Releases)

Versionierung: SemVer, CHANGELOG.md gepflegt.

Release: Tag + Release Notes + reproduzierbares Demo-Bundle.

Artefakte:

Python: dist/*.whl

Web: frontend/dist/

Docker: ghcr.io/<org>/<app>:<version>

7) Fachliche Korrektheit (Naturwissenschaften)

Einheiten strikt SI, Unsicherheiten angeben.

Dimensionsanalyse vor Merge.

Numerische Stabilität: Konditionszahl, Fehlerfortpflanzung, Sensitivitätstests.

Vergleich gegen Literatur-Kurven/Tabellen (digitized falls nötig).

Randomisierte/Monte-Carlo-Checks für robuste Aussagen.

Grenzfälle (low/high Re, dünn/dick, ideal/nicht-ideal) als Tests.

8) Benchmarks & Performance

Zielmetrik (Zeit/Speicher/Throughput) definieren.

Referenzdaten und Baseline dokumentieren.

benchmarks/ mit Skripten + Roh-Ergebnissen (results.json) committen.

Regressionen >5 %: PR blockiert bis geklärt.

9) PR-Checkliste (Agent-Self-Review)

 Recherche durchgeführt, Quellen in docs/research/... & CITATIONS.bib.

 Fachliche Aussagen/Parameter mit Literatur belegt.

 README.md vollständig aktualisiert (Install, Build, Quickstart, Validierung).

 GUI benutzerfreundlich, a11y-Checks, Onboarding vorhanden.

 Lint/Typecheck/Tests grün, Abdeckung ausreichend.

 Benchmarks unverändert oder verbessert.

 Logging/Fehlermeldungen hilfreich, keine Secrets.

 SemVer-Bump + CHANGELOG.

 Beispiel-Pipelines/Notebooks laufen und sind aktuell.

 Code-Kommentare + Docstrings aussagekräftig.

10) Zitieren & Dokumentieren

BibTeX-Pflicht: docs/CITATIONS.bib.

In Code/Notebooks: Kurzverweis (Autor Jahr, Kap. X / Gl. Y) und DOI/URL in den Notes.

Replizierbare Zahlen: Quelle + Seiten/Tabellen-ID angeben.

Abbildungen aus Papers nur mit rechtlich zulässiger Nutzung (oder nachzeichnen und kennzeichnen).

11) Qualität, Sicherheit, Datenschutz

Eingabedaten validieren, schemabasiert (Pydantic/Zod).

PII/Proprietäre Daten nur nach Freigabe; Maskierung/Anonymisierung wo nötig.

Lizenz-Compliance (SPDX), Third-Party-Liste pflegen.

Determinismus für wissenschaftliche Ergebnisse (Seeds, Versions-Pins).

12) Definition of Done (DoD)

Ein Task ist fertig, wenn:

Recherche-Dossier + Quellen vorhanden und geprüft.

Fachliche Korrektheit demonstriert (Validierungsnotebook/Tests).

Code inkl. Tests, Linting, Typisierung, Benchmarks grün.

README.md aktualisiert, GUI benutzerfreundlich, Build/Release lauffähig.

Changelog, Version, Artefakte veröffentlicht.

13) Beispiel-Kommandos & Skripte
# Lint & Typecheck
ruff check . && mypy . && eslint . --max-warnings=0

# Tests
pytest -q && vitest run

# Benchmarks
python benchmarks/run_all.py --save results.json

# README-Sync
python scripts/sync_readme.py

# Build
uv build || poetry build
pnpm install && pnpm build
docker build -t org/app:$(git describe --tags) .

14) Vorlagen

Commit-Nachricht

feat(gui): intuitiver Parameterdialog + Validierung
- Literatur: Müller2023 (Kap. 4.2), Smith2022 (Eq. 7)
- README aktualisiert (Quickstart, Screenshots)
- Tests: gui_validation_test.py hinzugefügt


PR-Beschreibung

### Ziel
Benutzerfreundliche GUI für <Feature> mit validierter Parametereingabe.

### Recherche
Siehe /docs/research/<feature-id>/summary.md (Müller2023; Smith2022)

### Änderungen
- Neue API ...
- GUI-Flow ...
- Validierung ...
- README/CHANGELOG aktualisiert

### Tests/Benchmarks
- pytest: 58 Tests, grün
- Benchmarks: -12% Laufzeit

### DoD
Alle Punkte erfüllt (Checkliste oben)

15) Durchsetzung

Diese Regeln sind verbindlich.

CI blockiert Merges bei fehlender Recherche, fehlgeschlagenen Tests, veralteter README oder GUI-a11y-Verstößen.

Abweichungen nur per schriftlicher Ausnahme mit fachlicher Begründung und Risikoanalyse.
