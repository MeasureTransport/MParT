import sys
from .pympart import *

kokkos_init = Initialize(dict())
sys.modules[__name__] = sys.modules['mpart']
