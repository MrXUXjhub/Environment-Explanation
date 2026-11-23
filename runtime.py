"""CPU可复现配置，以及macOS下现有scikit-learn OpenMP运行库的复用。"""
import os
import sys
from pathlib import Path


def configure():
    root = Path(__file__).resolve().parent
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS']:
        os.environ.setdefault(key, '2')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.cache' / 'matplotlib'))
    os.environ.setdefault('NUMBA_CACHE_DIR', str(root / '.cache' / 'numba'))
    # dyld需在启动Python之前读取此变量；只对普通脚本入口重新启动一次。
    if sys.platform == 'darwin' and os.environ.get('PAPER5_OPENMP_READY') != '1':
        libdir = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/sklearn/.dylibs'
        if (libdir / 'libomp.dylib').is_file() and sys.argv[0].endswith('.py'):
            env = dict(os.environ)
            env['PAPER5_OPENMP_READY'] = '1'
            env['DYLD_LIBRARY_PATH'] = str(libdir) + (':' + env['DYLD_LIBRARY_PATH'] if env.get('DYLD_LIBRARY_PATH') else '')
            os.execve(sys.executable, [sys.executable, *sys.argv], env)
