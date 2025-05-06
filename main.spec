# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
a = Analysis(
    ['main.py'],
    pathex=['D:/Code/Python/LicensePlate'],
    binaries=[],
    datas=[
    ('model/LicensePlate/Detect_Plate.pt', 'model/LicensePlate'),
    ('model/OCR2/ocr2.pt', 'model/OCR2'),
    ('function/helper.py', 'function'),
    ('function/utils_rotate.py', 'function'),
    ],

    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico',
)
