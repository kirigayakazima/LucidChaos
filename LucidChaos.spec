# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = [
    ('lucid_chaos/templates', 'lucid_chaos/templates'),
    ('lucid_chaos/config.json', 'lucid_chaos'),
]
binaries = []
hiddenimports = [
    'lucid_chaos',
    'lucid_chaos.ui',
    'lucid_chaos.ui.main_window',
    'lucid_chaos.ui.home_interface',
    'lucid_chaos.ui.settings_interface',
    'lucid_chaos.ui.toolbox_interface',
    'lucid_chaos.ui.worker',
    'lucid_chaos.core',
    'lucid_chaos.core.window',
    'lucid_chaos.core.capture',
    'lucid_chaos.core.input',
    'lucid_chaos.core.vision',
    'lucid_chaos.core.features',
    'lucid_chaos.core.text_match',
    'lucid_chaos.core.stuck_guard',
    'lucid_chaos.logic',
    'lucid_chaos.logic.node_engine',
    'lucid_chaos.logic.battle_manager',
    'lucid_chaos.logic.path_planner',
    'lucid_chaos.logic.story_pusher',
    'lucid_chaos.router',
    'lucid_chaos.router.router',
    'lucid_chaos.router.scene_types',
    'lucid_chaos.router.scene_pool',
    'lucid_chaos.router.evaluators',
    'lucid_chaos.state',
    'lucid_chaos.state.observer',
    'lucid_chaos.state.screen_state',
    'lucid_chaos.fsm',
    'lucid_chaos.fsm.engine',
    'lucid_chaos.fsm.orchestrator',
    'win32gui',
    'win32ui',
    'win32con',
    'win32api',
    'win32process',
]

# Collect all resources and models from rapidocr_onnxruntime and qfluentwidgets
for pkg in ['rapidocr_onnxruntime', 'qfluentwidgets', 'onnxruntime']:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)

a = Analysis(
    ['run_gui.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'torch'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LucidChaos',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LucidChaos',
)
