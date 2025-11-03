#!/usr/bin/env python3
"""
🧪 TIFF Simulator V4.1 - App Tester
====================================

Testet ob alle Module korrekt importiert werden können.
Laufe dies VOR dem Build um sicherzugehen, dass alles funktioniert!

Usage:
    python TEST_APP.py
"""

import sys

print("=" * 70)
print("🧪 TIFF SIMULATOR V4.1 - MODULE TEST")
print("=" * 70)

tests_passed = 0
tests_failed = 0

def test_import(module_name, description):
    """Test single import."""
    global tests_passed, tests_failed
    try:
        __import__(module_name)
        print(f"   ✓ {description}")
        tests_passed += 1
        return True
    except ImportError as e:
        print(f"   ❌ {description}")
        print(f"      Error: {e}")
        tests_failed += 1
        return False

# Test Core Dependencies
print("\n📦 Core Dependencies:")
test_import("numpy", "NumPy")
test_import("PIL", "Pillow (PIL)")
test_import("tqdm", "tqdm")

# Test ML Dependencies
print("\n🤖 Machine Learning:")
test_import("sklearn", "scikit-learn")
test_import("joblib", "joblib")
test_import("scipy", "scipy")
test_import("scipy.stats", "scipy.stats")

# Test Visualization
print("\n📊 Visualization:")
test_import("matplotlib", "matplotlib")
test_import("matplotlib.pyplot", "matplotlib.pyplot")
test_import("openpyxl", "openpyxl")

# Test GUI
print("\n🖼️ GUI:")
test_import("tkinter", "tkinter")
try:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    root.destroy()
    print("   ✓ tkinter window test")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ tkinter window test")
    print(f"      Error: {e}")
    tests_failed += 1

# Test Project Modules
print("\n🔬 Project Modules:")
test_import("tiff_simulator_v3", "tiff_simulator_v3.py")
test_import("metadata_exporter", "metadata_exporter.py")
test_import("batch_simulator", "batch_simulator.py")
test_import("rf_trainer", "rf_trainer.py")
test_import("track_analysis", "track_analysis.py")
test_import("adaptive_rf_trainer", "adaptive_rf_trainer.py")
test_import("diffusion_label_utils", "diffusion_label_utils.py")

# Test GUI Main
print("\n🎮 GUI Main:")
try:
    import tiff_simulator_gui_v4
    print("   ✓ tiff_simulator_gui_v4.py")
    tests_passed += 1
except ImportError as e:
    print(f"   ❌ tiff_simulator_gui_v4.py")
    print(f"      Error: {e}")
    tests_failed += 1

# Test Key Functions
print("\n🔑 Key Functions:")
try:
    from adaptive_rf_trainer import PolygradEstimator, quick_train_adaptive_rf
    print("   ✓ Adaptive RF Trainer API")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ Adaptive RF Trainer API")
    print(f"      Error: {e}")
    tests_failed += 1

try:
    from track_analysis import TrackAnalysisOrchestrator, TrackMateXMLParser
    print("   ✓ Track Analysis API")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ Track Analysis API")
    print(f"      Error: {e}")
    tests_failed += 1

try:
    from rf_trainer import RandomForestTrainer, RFTrainingConfig
    print("   ✓ RF Trainer API")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ RF Trainer API")
    print(f"      Error: {e}")
    tests_failed += 1

# Summary
print("\n" + "=" * 70)
print(f"📊 TEST RESULTS: {tests_passed} passed, {tests_failed} failed")
print("=" * 70)

if tests_failed == 0:
    print("\n✅ ALLE TESTS BESTANDEN!")
    print("   Du kannst jetzt die App builden:")
    print("   python BUILD_APP.py")
    print()
    sys.exit(0)
else:
    print(f"\n❌ {tests_failed} TESTS FEHLGESCHLAGEN!")
    print("   Installiere fehlende Dependencies:")
    print("   pip install -r requirements.txt")
    print()
    sys.exit(1)
