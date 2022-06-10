import sys
from .pympart import *

kokkos_init = KokkosInit({})
sys.modules[__name__] = sys.modules['mpart']
