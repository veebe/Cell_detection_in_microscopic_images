from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules('numba')

hiddenimports += [
    'numba.core.types.old_scalars',
    'numba.core.types.containers',
    'numba.core.types.misc',
    'numba.core.types.common',
    'numba.core.dispatcher',
    'numba.core.registry',
    'numba.core.event'
]

datas = collect_data_files('numba')