#!/usr/bin/env python3
"""Single-thread BLAS established before NumPy import; maximum two requested."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from replaystudy.stages import main
if __name__=='__main__': main()
