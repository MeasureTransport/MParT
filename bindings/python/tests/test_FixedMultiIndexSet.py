# FixedMultiIndexSet test

import mpart
import numpy as np

def test_max_degrees():
    multis = np.array([[0, 1], [2, 0]])
    mset = mpart.MultiIndexSet(multis)
    fixed_mset = mset.fix(True)
    assert np.all(fixed_mset == [1,2])
