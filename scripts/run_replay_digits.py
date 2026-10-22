"""Fresh-process entry point: limit numerical-library threads before NumPy import."""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OPENBLAS_DEFAULT_NUM_THREADS', 'OMP_NUM_THREADS',
             'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

from functional_circuits.replay_study import main

if __name__ == '__main__':
    main()
