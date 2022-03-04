import sys
from .pympart import *

sys.modules[__name__] = sys.modules['mpart']
