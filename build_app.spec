# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for TIFF Simulator V4.1
# Updated: November 2025 - Inkl. Track Analysis & Adaptive RF Training

import os

block_cipher = None

a = Analysis(
    ['tiff_simulator_gui_v4.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Core Simulator Modules
        ('tiff_simulator_v3.py', '.'),
        ('metadata_exporter.py', '.'),
        ('batch_simulator.py', '.'),

        # NEW V4.1: Track Analysis & RF Training
        ('track_analysis.py', '.'),
        ('rf_trainer.py', '.'),
        ('rf_training_session.py', '.'),
        ('adaptive_rf_trainer.py', '.'),
        ('diffusion_label_utils.py', '.'),

        # Extended Documentation for Desktop Bundle
        ('README.md', '.'),
        ('QUICKSTART.md', '.'),
        ('CHANGELOG.md', '.'),
        ('PHYSICS_VALIDATION.md', '.'),
        ('BATCH_MODE_GUIDE.md', '.'),
        ('RF_USAGE_GUIDE.md', '.'),
        ('TRACK_ANALYSIS_GUIDE.md', '.'),
        ('ADAPTIVE_RF_GUIDE.md', '.'),
        ('SETUP_GUIDE.md', '.'),
    ],
    hiddenimports=[
        # GUI & Basics
        'PIL._tkinter_finder',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.ttk',

        # NumPy
        'numpy',
        'numpy.core',
        'numpy.core._methods',
        'numpy.lib.format',

        # Scikit-Learn (RF Training)
        'sklearn',
        'sklearn.ensemble',
        'sklearn.ensemble._forest',
        'sklearn.tree',
        'sklearn.tree._tree',
        'sklearn.utils',
        'sklearn.utils._typedefs',
        'sklearn.neighbors',
        'sklearn.neighbors._partition_nodes',

        # Joblib (Model Saving/Loading)
        'joblib',
        'joblib.externals.loky.backend.context',

        # SciPy (MSD Analysis)
        'scipy',
        'scipy.stats',
        'scipy.stats._stats_py',
        'scipy.special',
        'scipy.special._ufuncs',

        # Standalone RF Training Session
        'rf_training_session',

        # Matplotlib (Plotting)
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.figure',
        'matplotlib.backends',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.tkagg',
        'matplotlib.backends._tkagg',
        'matplotlib.backends._backend_tk',


        # OpenPyXL (Excel Export)
        'openpyxl',
        'openpyxl.cell',
        'openpyxl.styles',
        'openpyxl.worksheet',
        'openpyxl.workbook',

        # XML Parsing
        'xml.etree.ElementTree',

        # Other
        'tempfile',
        'shutil',
        'pathlib',
        'dataclasses',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused packages
        'IPython',
        'notebook',
        'pytest',
        'sphinx',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TIFF_Simulator_V4.1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # WICHTIG: True für Debugging! (zeigt Fehler)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico' if os.path.exists('app_icon.ico') else None,
)
