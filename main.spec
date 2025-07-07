# -*- mode: python ; coding: utf-8 -*-
import os
import sys

# Agregar el directorio del proyecto al path
project_dir = os.path.abspath('.')
sys.path.insert(0, project_dir)

# Definir los archivos de datos manualmente
datas = [
    ('agente', 'agente'),
    ('simulador', 'simulador'),
    ('simulador/data', 'simulador/data'),  # Incluir el directorio data del simulador
]

# Buscar todos los archivos .pkl en el proyecto
import glob
pkl_files = glob.glob('**/*.pkl', recursive=True)
for pkl_file in pkl_files:
    # Añadir cada archivo .pkl encontrado
    datas.append((pkl_file, os.path.dirname(pkl_file) if os.path.dirname(pkl_file) else '.'))

# Agregar específicamente archivos del simulador que pueden faltar
simulador_files = [
    'simulador/packages/classes/mejor_modelo_dnn_v2.pkl',
    'simulador/packages/classes/scaler_v2.pkl',
    'simulador/packages/classes/*.pkl',
]

for pattern in simulador_files:
    if '*' in pattern:
        files = glob.glob(pattern)
        for file in files:
            if os.path.exists(file):
                datas.append((file, os.path.dirname(file)))
    else:
        if os.path.exists(pattern):
            datas.append((pattern, os.path.dirname(pattern)))

# Agregar archivos específicos de stable_baselines3
try:
    import stable_baselines3
    sb3_path = os.path.dirname(stable_baselines3.__file__)
    version_file = os.path.join(sb3_path, 'version.txt')
    if os.path.exists(version_file):
        datas.append((version_file, 'stable_baselines3'))
    
    # Incluir todos los archivos de datos de stable_baselines3
    for root, dirs, files in os.walk(sb3_path):
        for file in files:
            if file.endswith(('.txt', '.json', '.yaml', '.yml')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, os.path.dirname(sb3_path))
                datas.append((full_path, rel_path))
except ImportError:
    pass

# Hidden imports - todos los módulos que PyInstaller debe incluir
hiddenimports = [
    # Módulos del proyecto - agente
    'agente',
    'agente.main_agent',
    'agente.classes',
    'agente.classes.local_agent',
    'agente.classes.base_agent',
    'agente.classes.sim_agent',
    'agente.custom_env',
    'agente.modules',
    'agente.modules.model_manager',
    'agente.modules.plotter',
    
    # Módulos del proyecto - simulador
    'simulador',
    'simulador.main_sim',
    'simulador.ejecutar_simulacion',
    'simulador.calcular_estadisticas',
    'simulador.packages',
    'simulador.packages.classes',
    'simulador.packages.classes.OLT_DNN',
    'simulador.packages.classes.OLT_DRL',
    'simulador.packages.classes.OLT',
    'simulador.packages.classes.TrafficGenerators',
    'simulador.packages.configuration',
    
    # Dependencias externas básicas
    'numpy',
    'pandas',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends',
    'matplotlib.backends.backend_agg',
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.scrolledtext',
    
    # Dependencias de ML (básicas)
    'stable_baselines3',
    'stable_baselines3.ppo',
    'stable_baselines3.common',
    'stable_baselines3.common.vec_env',
    'stable_baselines3.common.vec_env.dummy_vec_env',
    'stable_baselines3.common.callbacks',
    'stable_baselines3.common.env_util',
    'stable_baselines3.common.policies',
    'stable_baselines3.common.utils',
    'stable_baselines3.common.logger',
    'stable_baselines3.common.noise',
    'stable_baselines3.common.buffers',
    
    'torch',
    'torch.nn',
    'torch.optim',
    
    'sklearn',
    'sklearn.preprocessing',
    'sklearn.neural_network',
    'scipy',
    'scipy.stats',
    'scipy.stats.t',
    'scipy.stats.norm',
    'joblib',
    'joblib.numpy_pickle',
    
    'gymnasium',
    'gym',
    'gym.spaces',
    
    'simpy',
    'bitarray',
    'bitarray.util',
    
    # Dependencias de pkg_resources y setuptools
    'pkg_resources',
    'setuptools',
    'setuptools.extern',
    'wheel',
    'wheel.metadata',
    'tomli',
    'jaraco',
    'jaraco.text',
    'jaraco.functools',
    'jaraco.collections',
    'more_itertools',
    'importlib_metadata',
    'zipp',
    
    # Otros módulos que pueden ser necesarios
    'threading',
    'queue',
    'argparse',
    'datetime',
    'sys',
    'os',
]

a = Analysis(
    ['main.py'],
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pip',
        # 'wheel',  # Removido porque causa conflictos con setuptools
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SimDeepPON-RL',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Cambiar a True temporalmente para ver errores de importación
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
    name='SimDeepPON-RL'
)
