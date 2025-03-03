# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_submodules, collect_all

block_cipher = None

hookspath = ['hooks'] 

backend_imports = collect_submodules('backend')
frontend_imports = collect_submodules('frontend')
numba_imports = collect_submodules('numba')
stardist_imports = collect_submodules('stardist')

numba_datas, numba_binaries, numba_hiddenimports = collect_all('numba')
stardist_datas, stardist_binaries, stardist_hiddenimports = collect_all('stardist')

hidden_imports = backend_imports + frontend_imports + numba_imports + stardist_imports + numba_hiddenimports + stardist_hiddenimports

project_root = os.path.abspath(os.path.dirname('program/main.py'))

datas = [
    (os.path.join(project_root, 'icons'), 'icons'),
] + numba_datas + stardist_datas

a = Analysis(
    ['program/main.py'],
    pathex=['program'],
    binaries=numba_binaries + stardist_binaries,
    datas=datas,
    hiddenimports=hidden_imports + ['numba.core.types.old_scalars'],
    hookspath=hookspath,
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


tensorflow_modules = []
for (dest, source, kind) in a.datas:
    if dest.startswith('tensorflow') or dest.startswith('keras') or dest.startswith('numba'):
        tensorflow_modules.append((dest, source, kind))

for item in tensorflow_modules:
    a.datas.remove(item)

a.datas.extend(tensorflow_modules)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [], 
    exclude_binaries=True,  
    name='CellDetect',
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
    name='CellDetect',
)