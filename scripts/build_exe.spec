# -*- mode: python ; coding: utf-8 -*-
import os
PROJ_ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))

block_cipher = None

a = Analysis(
    [os.path.join(PROJ_ROOT, 'src', 'main.py')],
    pathex=[os.path.join(PROJ_ROOT, 'src')],
    binaries=[],
    datas=[
        (os.path.join(PROJ_ROOT, 'src', 'resources'), 'resources'),
    ],
    hiddenimports=[
        'faster_whisper',
        'ctranslate2',
        'huggingface_hub',
        'tokenizers',
        'av',
        'av.audio',
        'av.audio.resampler',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'sounddevice',
        'sounddevice._portaudio',
        'pyaudiowpatch',
        'numpy',
        'scipy',
        'scipy.signal',
        'soundfile',
        'core.audio_recorder',
        'core.transcriber',
        'core.session_manager',
        'core.echo_gate',
        'core.echo_diagnostics',
        'core.silence_detector',
        'gui.system_tray',
        'gui.main_window',
        'gui.settings_dialog',
        'gui.theme',
        'utils.config',
        'utils.logger',
        'utils.resource_path',
        'version',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Cadence',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
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
    upx=False,
    upx_exclude=[],
    name='Cadence',
)
