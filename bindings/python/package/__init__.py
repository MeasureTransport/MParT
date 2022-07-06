import sys
from .pympart import *

kokkos_init = Initialize({})
sys.modules[__name__] = sys.modules['mpart']
