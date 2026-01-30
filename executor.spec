# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(
    ['src\\executor.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'cv2',
        'scipy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'pytz',
        'PIL',
        'jinja2',
        'markupsafe',
        'joblib',
        'filelock',
        'fsspec',
        'networkx',
        'sympy',
        'mpmath',
        'dateutil',
        'typing_extensions',
        'zstandard',
        'packaging',
        'tzdata',
        'threadpoolctl',
        # Common sklearn hidden sub-dependencies
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree',
        'sklearn.tree._utils'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    optimize=0,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='executor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='executor'
)
