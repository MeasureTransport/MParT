import sys
from .pympart import *
import multiprocessing

ncpus = multiprocessing.cpu_count()
kokkos_init = Initialize({"kokkos-threads": ncpus})
sys.modules[__name__] = sys.modules['mpart']
