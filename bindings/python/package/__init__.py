import sys
from .pympart import *

kokkos_init = Initialize(dict())
sys.modules[__name__] = sys.modules['mpart']

try:
    from .torch import *
    mpart_has_torch = True
except ImportError:
    mpart_has_torch = False

